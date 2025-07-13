import json
import os
from dotenv import load_dotenv
import asyncio
import aiohttp
from openai import OpenAI
import logging

logger = logging.getLogger("pipeline")

async def llm_generate(prompt, system, model, temp=0.1, timeout=120, format=None, llm_setup=None, logprobs=False, top_logprobs=None):
    content = ""
    try:
        if llm_setup and llm_setup.get('use_openrouter', False):
            load_dotenv()
            key = os.getenv('OPENROUTER_API_KEY')
            if not key:
                raise ValueError("OPENROUTER_API_KEY not found")
            
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=key,
            )

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

            # Build API call parameters
            call_params = {
                "model": model,
                "messages": messages,
                "temperature": temp,
                "max_tokens": 4090,
                "timeout": timeout,
                "stream": False  # added for logprobs consistency
            }
            
            # Add logprobs
            if logprobs:
                call_params["logprobs"] = True
                if top_logprobs:
                    call_params["top_logprobs"] = top_logprobs
                logger.info(f"Requesting logprobs=True, top_logprobs={top_logprobs} from OpenRouter for model {model}")

            logger.debug(f"API call params: {call_params}")
            
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: client.chat.completions.create(**call_params)
            )
            
            # Inspect response to ensure complete capture of the response
            logger.info(f"OpenRouter API Response Details:")
            logger.info(f"  Model: {response.model}")
            logger.info(f"  Finish Reason: {response.choices[0].finish_reason}")
            logger.info(f"  Response ID: {response.id}")
            logger.info(f"  Usage: {response.usage}")
            
            # Check for problematic finish reasons
            if response.choices[0].finish_reason != "stop":
                logger.warning(f"Non-normal finish reason: {response.choices[0].finish_reason}")
            
            content = response.choices[0].message.content
            logger.debug(f"Response content length: {len(content) if content else 0}")
            
            # Return logprobs info if requested
            if logprobs:
                if response.choices[0].logprobs:
                    num_tokens = len(response.choices[0].logprobs.content) if response.choices[0].logprobs.content else 0
                    logger.info(f"Received logprobs for {num_tokens} tokens from OpenRouter")
                    logger.info(f"Logprobs available: {response.choices[0].logprobs is not None}")
                    if response.choices[0].logprobs.content:
                        logger.debug(f"First token logprob structure: {type(response.choices[0].logprobs.content[0])}")
                    return content, response.choices[0].logprobs
                else:
                    logger.warning(f"No logprobs received from OpenRouter despite requesting them")
                    logger.warning(f"Model: {model}, Finish reason: {response.choices[0].finish_reason}")
                    logger.warning(f"Response choice structure: {type(response.choices[0])}")
                    return content, None
            
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


async def llm_generate_with_retry(prompt, system, model, temp=0.1, timeout=120, format=None, llm_setup=None, logprobs=False, top_logprobs=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} for model {model}")
            result = await llm_generate(prompt, system, model, temp, timeout, format, llm_setup, logprobs, top_logprobs)
            
            # Check if we got a valid response
            if isinstance(result, tuple):
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


def openrouter_generate(prompt, model="google/gemini-2.0-flash-001", system="", temp=0.1, timeout=120, format=None):
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
    return result

