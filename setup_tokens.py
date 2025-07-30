"""
안전한 토큰 관리를 위한 유틸리티 함수들
"""

import os
import json
from pathlib import Path
from typing import Optional, Dict
import getpass

class SecureTokenManager:
    """
    안전한 토큰 관리 클래스
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.env_file = self.project_root / ".env"
        
    def load_env_file(self) -> Dict[str, str]:
        """
        .env 파일에서 환경 변수 로드
        """
        env_vars = {}
        
        if self.env_file.exists():
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        env_vars[key.strip()] = value.strip()
        
        return env_vars
    
    def get_token(self, token_name: str, prompt_if_missing: bool = True) -> Optional[str]:
        """
        토큰을 가져오는 함수
        
        우선순위:
        1. 환경 변수
        2. .env 파일
        3. 사용자 입력 (prompt_if_missing=True인 경우)
        """
        # 1. 환경 변수에서 확인
        token = os.getenv(token_name)
        if token:
            return token
        
        # 2. .env 파일에서 확인
        env_vars = self.load_env_file()
        token = env_vars.get(token_name)
        if token and token != f"your_{token_name.lower()}_here":
            return token
        
        # 3. 사용자 입력
        if prompt_if_missing:
            print(f"\n🔑 {token_name}이 설정되지 않았습니다.")
            print("다음 중 하나의 방법으로 설정할 수 있습니다:")
            print(f"1. 환경 변수: export {token_name}=your_token")
            print(f"2. .env 파일에 추가: {token_name}=your_token")
            print("3. 지금 입력하기")
            
            choice = input("지금 입력하시겠습니까? (y/n): ").lower()
            if choice == 'y':
                token = getpass.getpass(f"Enter {token_name}: ")
                
                # .env 파일에 저장할지 물어보기
                save_choice = input("이 토큰을 .env 파일에 저장하시겠습니까? (y/n): ").lower()
                if save_choice == 'y':
                    self.save_to_env_file(token_name, token)
                
                return token
        
        return None
    
    def save_to_env_file(self, token_name: str, token_value: str):
        """
        토큰을 .env 파일에 저장
        """
        env_vars = self.load_env_file() if self.env_file.exists() else {}
        env_vars[token_name] = token_value
        
        with open(self.env_file, 'w') as f:
            f.write("# 환경 변수 파일\n")
            f.write("# 이 파일은 Git에 업로드되지 않습니다\n\n")
            
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        
        print(f"✅ {token_name}이 .env 파일에 저장되었습니다.")
    
    def setup_github_auth(self) -> bool:
        """
        GitHub 인증 설정
        """
        token = self.get_token("GITHUB_TOKEN")
        
        if token:
            try:
                # GitHub CLI 인증
                import subprocess
                result = subprocess.run(
                    ["gh", "auth", "login", "--with-token"],
                    input=token,
                    text=True,
                    capture_output=True
                )
                
                if result.returncode == 0:
                    print("✅ GitHub CLI 인증이 완료되었습니다.")
                    return True
                else:
                    print(f"❌ GitHub CLI 인증 실패: {result.stderr}")
                    return False
                    
            except FileNotFoundError:
                print("❌ GitHub CLI가 설치되지 않았습니다.")
                print("설치 방법: https://cli.github.com/")
                return False
        
        return False
    
    def check_gitignore(self):
        """
        .gitignore에 .env 파일이 포함되어 있는지 확인
        """
        gitignore_file = self.project_root / ".gitignore"
        
        if gitignore_file.exists():
            with open(gitignore_file, 'r') as f:
                content = f.read()
                
            if ".env" not in content:
                print("⚠️  .gitignore에 .env가 추가되지 않았습니다.")
                add_choice = input(".gitignore에 .env를 추가하시겠습니까? (y/n): ").lower()
                
                if add_choice == 'y':
                    with open(gitignore_file, 'a') as f:
                        f.write("\n# 환경 변수 파일\n.env\n")
                    print("✅ .gitignore에 .env가 추가되었습니다.")
        else:
            print("❌ .gitignore 파일이 없습니다.")

def setup_project_tokens():
    """
    프로젝트 토큰 설정을 위한 대화형 함수
    """
    print("🔐 ViT Test 프로젝트 토큰 설정")
    print("=" * 50)
    
    token_manager = SecureTokenManager()
    
    # .gitignore 확인
    token_manager.check_gitignore()
    
    # GitHub 토큰 설정
    print("\n📱 GitHub 토큰 설정")
    if token_manager.setup_github_auth():
        print("GitHub 설정이 완료되었습니다.")
    else:
        print("GitHub 설정에 실패했습니다.")
    
    # 다른 토큰들 확인
    print("\n🤖 기타 토큰 확인")
    other_tokens = ["HUGGINGFACE_TOKEN", "OPENAI_API_KEY"]
    
    for token_name in other_tokens:
        token = token_manager.get_token(token_name, prompt_if_missing=False)
        if token:
            print(f"✅ {token_name}: 설정됨")
        else:
            print(f"⚠️  {token_name}: 설정되지 않음")
    
    print("\n🎉 토큰 설정이 완료되었습니다!")
    print("\n💡 다른 컴퓨터에서 사용하는 방법:")
    print("1. 이 프로젝트를 git clone")
    print("2. python setup_tokens.py 실행")
    print("3. 각 컴퓨터에서 개별적으로 토큰 입력")

if __name__ == "__main__":
    setup_project_tokens()
