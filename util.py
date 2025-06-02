import io
import base64
import requests
from PIL import Image


def fetch_image_for_pil(image_url: str) -> Image.Image | None:
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        img = Image.open(io.BytesIO(response.content))

        if img.mode in ('RGBA', 'LA', 'P'):
            img = img.convert('RGB')
        elif img.mode != 'RGB':
            print(f"이미지 '{image_url}' 모드({img.mode})를 RGB로 변환합니다.")
            img = img.convert('RGB')
        return img
    except requests.exceptions.RequestException as e:
        print(f"오류: URL에서 이미지를 가져오는 데 실패했습니다 ({image_url}): {e}")
        return None
    except IOError as e: # Pillow 관련 오류
        print(f"오류: 이미지 데이터를 PIL 객체로 처리하는 데 실패했습니다 ({image_url}): {e}")
        return None
    except Exception as e:
        print(f"오류: PIL 이미지 처리 중 알 수 없는 문제 ({image_url}): {e}")
        return None


def fetch_image_for_base64(image_url: str, model_name: str):
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(image_url, headers=headers, timeout=15)
        response.raise_for_status()

        img_bytes = response.content
        img = Image.open(io.BytesIO(img_bytes))
        image_type = Image.MIME.get(img.format)
        if not image_type:
            ext = image_url.split('.')[-1].lower()
            if ext == "jpg" or ext == "jpeg":
                image_type = "image/jpeg"
            elif ext == "png":
                image_type = "image/png"
            elif ext == "gif":
                image_type = "image/gif"
            elif ext == "webp":
                image_type = "image/webp"
            else:
                image_type = "image/jpeg"
                if img.format != 'JPEG':
                    img = img.convert('RGB')
                    temp_io = io.BytesIO()
                    img.save(temp_io, format='JPEG')
                    img_bytes = temp_io.getvalue()
        base64_encoded_data = base64.b64encode(img_bytes).decode('utf-8')
        if 'gpt' in model_name:
            return f"data:{image_type};base64,{base64_encoded_data}", image_type
        elif 'claude' in model_name:
            return {
                "type": "base64",
                "media_type": image_type,
                "data": base64_encoded_data,
            }, None
        else:
            return None, None
    except requests.exceptions.RequestException as e:
        print(f"오류: URL에서 이미지를 가져오는 데 실패했습니다 ({image_url}): {e}")
        return None, None
    except IOError as e:
        print(f"오류: 이미지 데이터를 처리하는 데 실패했습니다 ({image_url}): {e}")
        return None, None
    except Exception as e:
        print(f"오류: 이미지 인코딩 중 알 수 없는 문제 ({image_url}): {e}")
        return None, None
