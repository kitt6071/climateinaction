import json
import os
from dotenv import load_dotenv
import asyncio
import aiohttp
from openai import OpenAI
import logging

logger = logging.getLogger("pipeline")

async def llm_generate(prompt, system, model, temp=0.1, timeout=120, format=None, llm_setup=None):
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

            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temp,
                    max_tokens=4090,
                    timeout=timeout
                )
            )
            content = response.choices[0].message.content
            
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
                except aiohttp.ClientResponseError as http_err:
                    logger.error(f"Ollama HTTP error: {http_err.status} for {model}")
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
        content = ""
    
    return strip_markdown_json(content)


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
            "timeout": timeout
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
        
        response = client.chat.completions.create(**call_params)
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
