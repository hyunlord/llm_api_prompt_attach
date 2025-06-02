from google import genai
import anthropic
import openai

from util import fetch_image_for_pil, fetch_image_for_base64
from read_api_key import read_api_key
from image_api import image_api


def get_image_relevance_scores(query: str, image_urls: list[str], api_key: str, model_name: str) -> dict:
    results_scores = {}
    try:
        if 'gemini' in model_name:
            client = genai.Client(api_key=api_key)
        elif 'gpt' in model_name:
            client = openai.OpenAI(api_key=api_key)
        elif 'claude' in model_name:
            client = anthropic.Anthropic(api_key=api_key)
        else:
            raise ValueError("모델 이름이 잘못됨")
    except Exception as e:
        print(f"오류: Client 초기화 실패: {e}")
        for url in image_urls:
            results_scores[url] = f"Error: Client initialization failed ({type(e).__name__})"
        return {"scores": results_scores, "usage_metadata": None}

    api_usage_info = None
    images_for_api = []
    valid_image_urls = []
    print("이미지 객체 가져오는 중...")
    for image_url in image_urls:
        if 'gemini' in model_name:
            fetch_image = fetch_image_for_pil(image_url)
        else:
            fetch_image, _ = fetch_image_for_base64(image_url, model_name)
        if fetch_image:
            if 'gemini' in model_name:
                images_for_api.append(fetch_image)
            elif 'gpt' in model_name:
                images_for_api.append({"type": "image_url", "image_url": {"url": fetch_image, "detail": "auto"}})
            elif 'claude' in model_name:
                images_for_api.append({"type": "image", "source": fetch_image})
            else:
                raise ValueError("모델 이름이 잘못됨")
            valid_image_urls.append(image_url)
        else:
            results_scores[image_url] = "Error: Could not fetch or process image"

    if not images_for_api:
        print("API로 전송할 유효한 PIL 이미지가 없습니다.")
        return {"scores": results_scores, "usage_metadata": None}

    num_valid_images = len(valid_image_urls)
    print(f"{num_valid_images}개의 이미지를 API({model_name})로 전송합니다.")

    prompt_text = f"""당신은 이미지 분석가로서 다음 검색 쿼리와 **앞서 제공된 {num_valid_images}개의 이미지들** 각각의 관련성을 매우 신중하고 비판적으로 평가해야 합니다.
    검색 쿼리: "{query}"

    평가 지침:
    1.  점수 범위: 각 이미지에 대해 0.0점에서 100.0점 사이의 실수 점수(소수점 한두 자리까지 포함 가능)를 부여합니다.
    2.  점수 해석:
        - 0.0점: 쿼리와 전혀 관련이 없는 이미지.
        - 1.0점 ~ 20.0점: 쿼리와 거의 또는 매우 낮은 관련성을 가짐.
        - 20.1점 ~ 40.0점: 쿼리와 어느 정도 낮은 관련성을 가짐.
        - 40.1점 ~ 60.0점: 쿼리와 보통 수준의 관련성을 가짐 (애매하거나 일부만 부합).
        - 60.1점 ~ 80.0점: 쿼리와 상당히 높은 관련성을 가짐.
        - 80.1점 ~ 99.9점: 쿼리와 매우 높은 관련성을 가지며, 주요 요소들이 잘 부합함.
        - 100.0점: 쿼리의 모든 측면과 완벽하게 일치하는 이상적인 이미지 (매우 드문 경우에만 부여).
    3.  차등 평가: **제공된 순서대로** 각 이미지의 쿼리에 대한 관련성 수준을 면밀히 비교하고, 가능한 0.0점에서 100.0점까지의 점수 범위를 넓게 활용하여 각 이미지에 차별화된 점수를 부여해주십시오. 단순히 몇몇 특정 점수(예: 5.0, 50.0, 95.0)에 집중하지 마십시오.
    4.  획일적 점수 지양: 모든 이미지의 관련성이 정말로 모든 면에서 동일하다고 판단되는 극히 예외적인 경우를 제외하고는, 모든 이미지에 동일한 점수를 부여하는 것을 반드시 피해주십시오. 이미지 간의 미세한 차이라도 발견하여 점수에 반영하도록 노력해주십시오.

    출력 형식:
    오직 점수들만 쉼표(,)로 구분된 목록 형태로 응답해주십시오. 다른 어떤 설명, 인사, 추가 텍스트도 포함하지 마십시오.
    예시 (이미지가 3개일 경우, 제공된 순서대로): '92.5,55.0,15.75'

    이제, 앞서 제공된 {num_valid_images}개의 이미지를 위의 지침에 따라 평가하고 점수 목록을 반환해주십시오:
    """
    contents, messages = None, None
    if 'gemini' in model_name:
        contents = [prompt_text] + images_for_api
    else:
        message_content = [{"type": "text", "text": prompt_text}] + images_for_api
        messages = [{"role": "user", "content": message_content}]
    try:
        print("\nAPI 호출 중 ...")
        if 'gemini' in model_name:
            response = client.models.generate_content(
                 model=f"models/{model_name}",
                 contents=contents
            )
            api_usage_info = {
                "prompt_token_count": response.usage_metadata.prompt_token_count,
                "candidates_token_count": response.usage_metadata.candidates_token_count,
                "total_token_count": response.usage_metadata.total_token_count,
            }
            ai_response_text = response.text.strip()
        elif 'gpt' in model_name:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
            api_usage_info = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            ai_response_text = response.choices[0].message.content.strip()
        elif 'claude' in model_name:
            response = client.messages.create(
                model=model_name,
                messages=messages
            )
            api_usage_info = {
                "prompt_tokens": response.usage.input_tokens,  # Anthropic은 input_tokens
                "completion_tokens": response.usage.output_tokens,  # Anthropic은 output_tokens
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,  # 직접 계산 필요시
            }
            ai_response_text = response.content[0].text.strip()
        else:
            raise ValueError("모델 이름이 잘못됨")
        print(f"API 토큰 사용량: {api_usage_info}")
        score_strings = [s.strip() for s in ai_response_text.split(',')]

        if len(score_strings) != num_valid_images:
            print(f" 경고: AI가 반환한 점수의 개수({len(score_strings)})가 이미지 개수({num_valid_images})와 다릅니다.")
            for i in range(num_valid_images):
                results_scores[valid_image_urls[i]] = "Error: Score count mismatch from AI"
            return {"scores": results_scores, "usage_metadata": api_usage_info}

        for i in range(num_valid_images):
            url = valid_image_urls[i]
            try:
                score = float(score_strings[i])
                if not (0.0 <= score <= 100.0):
                    results_scores[url] = f"Error: Invalid score range from AI ({score})"
                else:
                    results_scores[url] = round(score, 2)
            except ValueError:
                results_scores[url] = f"Error: Non-numeric AI response ('{score_strings[i]}')"
            except IndexError:
                results_scores[url] = "Error: Missing score from AI"
    except Exception as e:
        print(f"오류: API 호출 중 문제 발생 ({model_name}): {e}")
        for url in valid_image_urls:
            if url not in results_scores:
                 results_scores[url] = f"Error: API call failed ({type(e).__name__})"
        error_message_lower = str(e).lower()
        if "api_key_invalid" in error_message_lower or "api key not valid" in error_message_lower:
             print("API 키가 유효한지 확인해주세요.")
        if "resource_exhausted" in error_message_lower:
            print("API 할당량이 초과되었을 수 있습니다. Google Cloud Console에서 할당량을 확인하세요.")
        if "was not found" in error_message_lower and model_name in str(e):
            print(f"모델명 '{model_name}'을 찾을 수 없습니다. 유효한 멀티모달 모델명인지, 'models/' 접두사가 필요한지 확인해주세요.")
    return {"scores": results_scores, "usage_metadata": api_usage_info}


if __name__ == '__main__':
    query = input()
    count = int(input())
    print(query, count)
    image_urls_origin = image_api(query, count, 'origin')
    #image_urls_test = image_api_test(query, count, 'test')
    gemini_api_key, openai_api_key, claude_api_key = read_api_key()

    result_gemini = get_image_relevance_scores(query, image_urls_origin, gemini_api_key, "gemini-2.0-flash-lite-001")
    result_openai = get_image_relevance_scores(query, image_urls_origin, openai_api_key, "gpt-4.1-mini-2025-04-14")
    result_anthropic = get_image_relevance_scores(query, image_urls_origin, claude_api_key, "claude-3-5-sonnet-20240620")

    for i, j, k in zip(result_gemini['scores'].items(), result_openai['scores'].items(), result_anthropic['scores'].items()):
        print(i[0], '|||', i[1], '|||', j[1], '|||', k[1])

