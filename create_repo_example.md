# GitHub MCP를 통한 Repository 생성 가이드

## 설정 완료 후 사용 방법

1. **VS Code Command Palette 열기** (`Ctrl+Shift+P`)

2. **MCP를 통한 GitHub 작업**
   - "GitHub: Create Repository" 명령 실행
   - 또는 채팅에서 GitHub MCP 서버에 요청

## 채팅을 통한 Repository 생성 예시

```
새로운 GitHub repository를 생성해주세요:
- 이름: my-awesome-project
- 설명: My awesome new project
- 공개/비공개: public
- README.md 파일 포함
```

## CLI를 통한 대안 방법

만약 MCP 연결에 문제가 있다면, GitHub CLI를 사용할 수도 있습니다:

```bash
# GitHub CLI 설치 (Ubuntu/Debian)
sudo apt update
sudo apt install gh

# 인증
gh auth login

# Repository 생성
gh repo create my-project-name --public --description "My project description"

# 또는 현재 디렉토리를 기반으로 생성
gh repo create --source=. --public
```

## 프로젝트 초기화

Repository 생성 후:

```bash
# Git 초기화
git init

# README 파일 생성
echo "# My Project" > README.md

# 파일 추가 및 커밋
git add .
git commit -m "Initial commit"

# 원격 저장소 연결 (MCP가 자동으로 해줄 수도 있음)
git remote add origin https://github.com/USERNAME/REPO-NAME.git
git push -u origin main
```
