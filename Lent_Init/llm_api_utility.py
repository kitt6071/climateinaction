import json
import os
from dotenv import load_dotenv
import asyncio
import aiohttp
from openai import OpenAI
import logging

logger = logging.getLogger("pipeline")

async def llm_generate(prompt, system, model, temp=0.1, timeout=180, format=None, llm_setup=None, logprobs=False, top_logprobs=None, extra_body=None):
    content = ""
    try:
        if llm_setup and llm_setup.get('use_openrouter', False):
            load_dotenv()
            key = os.getenv('OPENROUTER_API_KEY')
            if not key:
                raise ValueError("OPENROUTER_API_KEY not found")
            
            if llm_setup.get('api_rate_limiter'):
                await llm_setup['api_rate_limiter'].async_wait()

            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ]
            
            # openrouter json handling is weird, just modify the system prompt
            if format and isinstance(format, dict):
                 sys_msg = f"{system}\n\nRespond ONLY with valid JSON matching this schema: {json.dumps(format)}"
                 messages[0]["content"] = sys_msg
            elif format == "json":
                 sys_msg = f"{system}\n\nRespond ONLY with valid JSON."
                 messages[0]["content"] = sys_msg

            call_params = {
                "model": model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": 16000,
                #"require_parameters": True,
                "stream": False  # added for logprobs consistency
            }
            
            if extra_body:
                call_params.update(extra_body)
            
            # Add logprobs
            if logprobs:
                call_params["logprobs"] = True
                if top_logprobs:
                    call_params["top_logprobs"] = top_logprobs
                logger.debug(f"Requesting logprobs from OpenRouter for model {model}")

            logger.debug(f"API call params: {call_params}")
            
            headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)) as session:
                async with session.post("https://openrouter.ai/api/v1/chat/completions", 
                                       headers=headers, json=call_params) as http_response:
                     cost = 0.0
                     
                     if llm_setup and llm_setup.get("track_metrics", False):
                         logger.debug(f"OpenRouter headers: {dict(http_response.headers)}")
                     
                     cost_headers = ['x-openrouter-cost', 'openrouter-cost', 'x-cost', 'cost']
                     for header_name in cost_headers:
                         if header_name in http_response.headers:
                             try:
                                 cost = float(http_response.headers[header_name])
                                 if llm_setup and llm_setup.get("track_metrics", False):
                                     logger.debug(f"Found cost in header '{header_name}': ${cost}")
                                 break
                             except (ValueError, TypeError):
                                 continue
                     
                     response_data = await http_response.json()
                     if http_response.status != 200:
                         raise Exception(f"OpenRouter API error {http_response.status}: {response_data}")
                     
                     if not response_data:
                         raise Exception(f"Empty response data from OpenRouter")
                     
                     class PaymentResponse:
                         def __init__(self, data, cost):
                             if not data:
                                 raise Exception("No data provided to PaymentResponse")
                             self.model = data.get("model", model)
                             self.choices = [PaymentChoice(data["choices"][0])]
                             usage_data = data.get("usage", {})
                             usage_data["cost"] = cost  # Add cost to usage
                             self.usage = PaymentUsage(usage_data)
                     
                     class PaymentChoice:
                         def __init__(self, choice_data):
                             self.message = PaymentMessage(choice_data["message"])
                             self.finish_reason = choice_data.get("finish_reason")
                             logprobs_data = choice_data.get("logprobs")
                             self.logprobs = PaymentLogprobs(logprobs_data) if logprobs_data else None
                     
                     class PaymentMessage:
                         def __init__(self, msg_data):
                             self.content = msg_data["content"]
                     
                     class PaymentLogprobs:
                         def __init__(self, logprobs_data):
                             if logprobs_data:
                                 self.content = logprobs_data.get("content", [])
                             else:
                                 self.content = []
                     
                     class PaymentUsage:
                         def __init__(self, usage_data):
                             self.prompt_tokens = usage_data.get("prompt_tokens", 0)
                             self.completion_tokens = usage_data.get("completion_tokens", 0)
                             self.total_tokens = usage_data.get("total_tokens", 0)
                             self.cost = usage_data.get("cost", 0.0)
                     
                     response = PaymentResponse(response_data, cost)
            
            # Log only important response details
            logger.debug(f"API Response: {response.model}, finish: {response.choices[0].finish_reason}, tokens: {response.usage.total_tokens}")
            
            # Check for problematic finish reasons
            if response.choices[0].finish_reason != "stop":
                logger.warning(f"Non-normal finish reason: {response.choices[0].finish_reason}")
            
            content = response.choices[0].message.content
            
            usage_data = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
                "cost": response.usage.cost if response.usage else 0.0
            }
            
            if llm_setup and llm_setup.get("track_metrics", False):
                logger.info(f"LLM Metrics - Model: {model}, Prompt: {usage_data['prompt_tokens']}, Completion: {usage_data['completion_tokens']}, Total: {usage_data['total_tokens']}, Cost: ${usage_data['cost']:.6f}")
            
            if logprobs:
                if response.choices[0].logprobs:
                    logger.debug(f"Received logprobs for {len(response.choices[0].logprobs.content) if response.choices[0].logprobs.content else 0} tokens")
                    if llm_setup and llm_setup.get("return_metrics", False):
                        return content, response.choices[0].logprobs, usage_data
                    else:
                        return content, response.choices[0].logprobs
                else:
                    logger.debug(f"No logprobs received from OpenRouter")
                    if llm_setup and llm_setup.get("return_metrics", False):
                        return content, None, usage_data
                    else:
                        logger.warning(f"Model: {model}, Finish reason: {response.choices[0].finish_reason}")
                        return content, None
            
            return content
            
        else:
            # ollama
            url = "http://localhost:11434/api/generate"
            payload = {
                "model": model,
                "prompt": prompt,
                "system": system,
                "stream": False,
                "options": {
                "temperature": temp,
                }
            }
            if format:
                payload["format"] = "json"

            if llm_setup and llm_setup.get('api_rate_limiter'):
                 await llm_setup['api_rate_limiter'].async_wait()

            async with aiohttp.ClientSession() as session:
                try:
                    async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                        resp.raise_for_status()
                        result = await resp.json()
                        content = result.get("response", "")
                        
                        # Log Ollama response details
                        logger.info(f"Ollama Response Details:")
                        logger.info(f"  Model: {model}")
                        logger.info(f"  Response length: {len(content) if content else 0}")
                        logger.info(f"  Done: {result.get('done', 'unknown')}")
                        
                except aiohttp.ClientResponseError as http_err:
                    logger.error(f"Ollama HTTP error: {http_err.status} for {model}")
                    logger.error(f"Response text: {await http_err.response.text() if http_err.response else 'No response'}")
                    if http_err.status == 429 and llm_setup and llm_setup.get('api_rate_limiter'):
                        llm_setup['api_rate_limiter'].handle_async_rate_limit()
                    content = "" 
                except asyncio.TimeoutError:
                    logger.error(f"Ollama timeout after {timeout}s for {model}")
                    content = ""
                except Exception as err:
                    logger.error(f"Ollama error: {err} for {model}")
                    content = ""

    except Exception as err:
        logger.error(f"LLM generate error for {model}: {err}")
        logger.error(f"Error type: {type(err)}")
        content = ""
    
    return strip_markdown_json(content)


def enable_metrics_tracking(llm_setup):
    """Enable metrics tracking for LLM calls with minimal code changes."""
    if not llm_setup:
        llm_setup = {}
    
    llm_setup["track_metrics"] = True
    llm_setup["return_metrics"] = True
    llm_setup["metrics_tracker"] = {
        "total_cost": 0.0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_calls": 0
    }
    return llm_setup


def log_metrics_summary(llm_setup, logger=None):
    if not logger:
        import logging
        logger = logging.getLogger("pipeline")
    
    if llm_setup and llm_setup.get("metrics_tracker"):
        metrics = llm_setup["metrics_tracker"]
        logger.info("--------LLM USAGE METRICS------")
        logger.info(f"Total Cost: ${metrics['total_cost']:.6f}")
        logger.info(f"Total Calls: {metrics['total_calls']}")
        logger.info(f"Total Prompt Tokens: {metrics['total_prompt_tokens']:,}")
        logger.info(f"Total Completion Tokens: {metrics['total_completion_tokens']:,}")
        logger.info(f"Total Tokens: {metrics['total_prompt_tokens'] + metrics['total_completion_tokens']:,}")
        if metrics['total_calls'] > 0:
            logger.info(f"Average Cost per Call: ${metrics['total_cost'] / metrics['total_calls']:.6f}")
            logger.info(f"Average Tokens per Call: {(metrics['total_prompt_tokens'] + metrics['total_completion_tokens']) / metrics['total_calls']:.1f}")
    else:
        logger.info("No metrics tracking enabled")


def extract_content_from_result(result):
    if isinstance(result, dict):
        return result.get("content", "")
    elif isinstance(result, tuple):
        return result[0] if result else ""
    else:
        return str(result) if result else ""


async def llm_generate_with_retry(prompt, system, model, temp=0.1, timeout=180, format=None, llm_setup=None, logprobs=False, top_logprobs=None, max_retries=3, extra_body=None):
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} for model {model}")
            result = await llm_generate(prompt, system, model, temp, timeout, format, llm_setup, logprobs, top_logprobs, extra_body)
            
            if isinstance(result, dict) and llm_setup and llm_setup.get("return_metrics", False):
                content = result.get("content", "")
                if content and content.strip():
                    logger.info(f"Success on attempt {attempt + 1}")
                    if llm_setup.get("metrics_tracker"):
                        usage = result.get("usage", {})
                        llm_setup["metrics_tracker"]["total_cost"] += usage.get("cost", 0.0)
                        llm_setup["metrics_tracker"]["total_prompt_tokens"] += usage.get("prompt_tokens", 0)
                        llm_setup["metrics_tracker"]["total_completion_tokens"] += usage.get("completion_tokens", 0)
                        llm_setup["metrics_tracker"]["total_calls"] += 1
                    return result
                else:
                    logger.warning(f"Empty content on attempt {attempt + 1}")
            elif isinstance(result, tuple):
                if len(result) == 3:
                    content, logprobs_info, usage = result
                    if content and content.strip():
                        logger.info(f"Success on attempt {attempt + 1}")
                        if llm_setup and llm_setup.get("metrics_tracker"):
                            llm_setup["metrics_tracker"]["total_cost"] += usage.get("cost", 0.0)
                            llm_setup["metrics_tracker"]["total_prompt_tokens"] += usage.get("prompt_tokens", 0)
                            llm_setup["metrics_tracker"]["total_completion_tokens"] += usage.get("completion_tokens", 0)
                            llm_setup["metrics_tracker"]["total_calls"] += 1
                        return result
                    else:
                        logger.warning(f"Empty content on attempt {attempt + 1}")
                else:
                    content, logprobs_info = result
                    if content and content.strip():
                        logger.info(f"Success on attempt {attempt + 1}")
                        return result
                    else:
                        logger.warning(f"Empty content on attempt {attempt + 1}")
            elif result and result.strip():
                logger.info(f"Success on attempt {attempt + 1}")
                return result
            else:
                logger.warning(f"Empty result on attempt {attempt + 1}")
                
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} failed: {e}")
            
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt  # Exponential backoff
            logger.info(f"Waiting {wait_time}s before retry...")
            await asyncio.sleep(wait_time)
    
    logger.error(f"All {max_retries} attempts failed for model {model}")
    return "" if not logprobs else ("", None)


def openrouter_generate(prompt, model="google/gemini-2.0-flash-001", system="", temp=0.1, timeout=180, format=None):
    load_dotenv()
    
    key = os.getenv('OPENROUTER_API_KEY')
    if not key:
        raise ValueError("OPENROUTER_API_KEY not found")
    
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
        )
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        call_params = {
            "model": model,
            "messages": messages,
            "temperature": temp,
            "timeout": timeout,
            "stream": False
        }
        
        if format:
            call_params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "response_schema",
                    "strict": True,
                    "schema": format
                }
            }
        
        logger.debug(f"Sync API call params: {call_params}")
        response = client.chat.completions.create(**call_params)
        
        # Log response details
        logger.info(f"Sync OpenRouter Response Details:")
        logger.info(f"  Model: {response.model}")
        logger.info(f"  Finish Reason: {response.choices[0].finish_reason}")
        logger.info(f"  Response ID: {response.id}")
        
        return response.choices[0].message.content
        
    except Exception as err:
        logger.exception(f"OpenRouter error: {err}")
        return ""


def strip_markdown_json(text):
    if text is None:
        return ""
    result = text.strip()
    if result.startswith("```json") and result.endswith("```"):
        result = result[7:-3].strip()
    elif result.startswith("```") and result.endswith("```"):
        result = result[3:-3].strip()
    
    import re
    
    if not (result.startswith('{') or result.startswith('[')):
        json_pattern = r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\])'
        matches = re.findall(json_pattern, result, re.DOTALL)
        if matches:
            result = max(matches, key=len)
    
    return result

