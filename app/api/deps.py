import os

import firebase_admin
from firebase_admin import auth, credentials
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

def _ensure_firebase_app() -> None:
    """
    Lazy init to avoid import-time crash when service account is missing.
    Configure via env:
      FIREBASE_SERVICE_ACCOUNT_PATH=.../service_account.json
    Defaults to `service_account.json` at repo root.
    """
    if firebase_admin._apps:
        return
    path = (os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH") or "service_account.json").strip()
    if not path or not os.path.exists(path):
        raise RuntimeError(
            "Missing Firebase service account file. Set FIREBASE_SERVICE_ACCOUNT_PATH or place `service_account.json` in repo root."
        )
    cred = credentials.Certificate(path)
    firebase_admin.initialize_app(cred)

security = HTTPBearer()

def get_current_user(token: HTTPAuthorizationCredentials = Depends(security)):
    try:
        _ensure_firebase_app()
        # 2. Xác thực token gửi lên từ Frontend
        decoded_token = auth.verify_id_token(token.credentials)
        
        # 3. Lấy tenant_id từ thông tin user (đã setup trước đó hoặc mặc định)
        # Lưu ý: Lúc test ban đầu có thể chưa có custom claims, ta tạm lấy uid làm tenant_id để test
        user_id = decoded_token['uid']
        tenant_id = decoded_token.get('tenant_id', user_id) 
        
        return {
            "uid": user_id,
            "tenant_id": tenant_id,
            "email": decoded_token.get('email')
        }
    except RuntimeError as e:
        # Configuration error (missing service account, etc.) should be a 500, not a 401.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token không hợp lệ hoặc đã hết hạn"
        )
