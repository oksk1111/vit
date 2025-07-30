#!/bin/bash

# 여러 컴퓨터에서 안전하게 작업하기 위한 설정 스크립트

echo "🔐 ViT Test 프로젝트 - 멀티 컴퓨터 설정"
echo "============================================"

# 1. SSH 키 설정 확인
echo ""
echo "📡 1. SSH 키 설정 확인"
if [ -f ~/.ssh/id_rsa.pub ] || [ -f ~/.ssh/id_ed25519.pub ]; then
    echo "✅ SSH 키가 이미 존재합니다."
    
    if [ -f ~/.ssh/id_ed25519.pub ]; then
        echo "🔑 Ed25519 공개 키:"
        cat ~/.ssh/id_ed25519.pub
    elif [ -f ~/.ssh/id_rsa.pub ]; then
        echo "🔑 RSA 공개 키:"
        cat ~/.ssh/id_rsa.pub
    fi
    
    echo ""
    echo "💡 이 공개 키를 GitHub에 등록하세요:"
    echo "   GitHub → Settings → SSH and GPG keys → New SSH key"
else
    echo "❌ SSH 키가 없습니다. 새로 생성하시겠습니까? (y/n)"
    read -r create_ssh
    
    if [ "$create_ssh" = "y" ]; then
        echo "이메일 주소를 입력하세요:"
        read -r email
        
        ssh-keygen -t ed25519 -C "$email"
        
        echo ""
        echo "✅ SSH 키가 생성되었습니다!"
        echo "🔑 공개 키:"
        cat ~/.ssh/id_ed25519.pub
        
        echo ""
        echo "💡 이 공개 키를 GitHub에 등록하세요:"
        echo "   GitHub → Settings → SSH and GPG keys → New SSH key"
    fi
fi

# 2. Git 설정 확인
echo ""
echo "⚙️ 2. Git 설정 확인"
git_name=$(git config --global user.name)
git_email=$(git config --global user.email)

if [ -z "$git_name" ] || [ -z "$git_email" ]; then
    echo "❌ Git 사용자 정보가 설정되지 않았습니다."
    
    if [ -z "$git_name" ]; then
        echo "이름을 입력하세요:"
        read -r name
        git config --global user.name "$name"
    fi
    
    if [ -z "$git_email" ]; then
        echo "이메일을 입력하세요:"
        read -r email
        git config --global user.email "$email"
    fi
    
    echo "✅ Git 사용자 정보가 설정되었습니다."
else
    echo "✅ Git 설정:"
    echo "   이름: $git_name"
    echo "   이메일: $git_email"
fi

# 3. 원격 저장소 URL을 SSH로 변경
echo ""
echo "🔗 3. 원격 저장소 SSH 설정"
current_url=$(git remote get-url origin)

if [[ $current_url == https* ]]; then
    echo "현재 HTTPS URL: $current_url"
    echo "SSH URL로 변경하시겠습니까? (권장) (y/n)"
    read -r change_to_ssh
    
    if [ "$change_to_ssh" = "y" ]; then
        # HTTPS URL을 SSH URL로 변환
        ssh_url=$(echo "$current_url" | sed 's|https://github.com/|git@github.com:|')
        git remote set-url origin "$ssh_url"
        echo "✅ 원격 저장소가 SSH URL로 변경되었습니다: $ssh_url"
    fi
else
    echo "✅ 이미 SSH URL을 사용 중입니다: $current_url"
fi

# 4. GitHub CLI 설치 확인
echo ""
echo "🛠️ 4. GitHub CLI 설치 확인"
if command -v gh &> /dev/null; then
    echo "✅ GitHub CLI가 설치되어 있습니다."
    
    # 인증 상태 확인
    if gh auth status &> /dev/null; then
        echo "✅ GitHub CLI 인증이 되어 있습니다."
    else
        echo "❌ GitHub CLI 인증이 필요합니다."
        echo "토큰으로 인증하시겠습니까? (y/n)"
        read -r auth_choice
        
        if [ "$auth_choice" = "y" ]; then
            echo "GitHub Personal Access Token을 입력하세요:"
            read -r -s token
            echo "$token" | gh auth login --with-token
            echo "✅ GitHub CLI 인증이 완료되었습니다."
        fi
    fi
else
    echo "❌ GitHub CLI가 설치되지 않았습니다."
    echo ""
    echo "설치 방법:"
    echo "  Ubuntu/Debian: sudo apt install gh"
    echo "  macOS: brew install gh"
    echo "  Windows: winget install GitHub.cli"
fi

# 5. 토큰 설정 스크립트 실행
echo ""
echo "🔑 5. 토큰 설정"
echo "Python 토큰 설정 스크립트를 실행하시겠습니까? (y/n)"
read -r run_python_setup

if [ "$run_python_setup" = "y" ]; then
    if command -v python3 &> /dev/null; then
        python3 setup_tokens.py
    elif command -v python &> /dev/null; then
        python setup_tokens.py
    else
        echo "❌ Python이 설치되지 않았습니다."
    fi
fi

echo ""
echo "🎉 설정이 완료되었습니다!"
echo ""
echo "📋 다음 단계:"
echo "1. SSH 키를 GitHub에 등록했는지 확인"
echo "2. git push/pull이 정상 작동하는지 테스트"
echo "3. 다른 컴퓨터에서도 동일한 설정 실행"
echo ""
echo "🔒 보안 팁:"
echo "- .env 파일은 절대 Git에 커밋하지 마세요"
echo "- SSH 키의 private key는 안전하게 보관하세요"
echo "- 토큰은 필요한 최소 권한만 부여하세요"
