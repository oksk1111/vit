# 🔐 멀티 컴퓨터 보안 토큰 관리 가이드

이 가이드는 여러 컴퓨터에서 안전하게 작업하면서 토큰을 관리하는 방법을 설명합니다.

## 🎯 목표

- ✅ 토큰이 Git 저장소에 노출되지 않도록 보호
- ✅ 여러 컴퓨터에서 동일한 프로젝트 작업 가능
- ✅ 최소 권한 원칙 적용
- ✅ 토큰 분실 시 빠른 대응 가능

## 🚀 빠른 설정 (신규 컴퓨터)

### 1단계: 프로젝트 클론

```bash
git clone https://github.com/oksk1111/vit_test.git
cd vit_test
```

### 2단계: 자동 설정 스크립트 실행

```bash
./setup_multi_computer.sh
```

이 스크립트는 다음을 자동으로 설정합니다:
- SSH 키 생성/확인
- Git 사용자 정보 설정
- GitHub CLI 설치 확인
- 토큰 설정 도움

### 3단계: 수동 토큰 설정 (필요시)

```bash
python3 setup_tokens.py
```

## 🔑 토큰 관리 방법

### 방법 1: 환경 변수 (권장)

각 컴퓨터의 쉘 프로필에 추가:

```bash
# ~/.bashrc 또는 ~/.zshrc에 추가
export GITHUB_TOKEN="ghp_your_token_here"
export HUGGINGFACE_TOKEN="hf_your_token_here"
```

**장점:**
- 시스템 전체에서 사용 가능
- Git에 노출되지 않음
- 영구적으로 설정됨

### 방법 2: .env 파일 (프로젝트별)

프로젝트 루트에 `.env` 파일 생성:

```bash
# .env 파일 (이 파일은 .gitignore에 포함됨)
GITHUB_TOKEN=ghp_your_token_here
HUGGINGFACE_TOKEN=hf_your_token_here
```

**장점:**
- 프로젝트별 독립적 관리
- 다른 프로젝트와 토큰 분리
- 쉬운 백업/복원

### 방법 3: SSH 키 + GitHub CLI

```bash
# SSH 키 생성
ssh-keygen -t ed25519 -C "your_email@example.com"

# GitHub에 SSH 키 등록 후
gh auth login --web
```

**장점:**
- 토큰 없이도 Git 작업 가능
- 더 안전한 인증 방식
- 토큰 만료 걱정 없음

## 🛡️ 보안 모범 사례

### 1. 토큰 권한 최소화

GitHub Personal Access Token 생성 시 필요한 권한만 선택:

```
필수 권한:
✅ repo (전체 저장소 접근)
✅ workflow (GitHub Actions)

선택적 권한:
- read:user (사용자 정보 읽기)
- gist (Gist 관리)
```

### 2. 토큰 순환

```bash
# 토큰을 정기적으로 교체 (3-6개월)
# 기존 토큰 비활성화
# 새 토큰으로 모든 컴퓨터 업데이트
```

### 3. 토큰 분실 시 대응

```bash
# 1. GitHub에서 즉시 토큰 삭제
# 2. 새 토큰 생성
# 3. 모든 컴퓨터에서 토큰 업데이트
```

## 🖥️ 컴퓨터별 설정 예시

### 메인 개발 컴퓨터
```bash
# 모든 권한을 가진 토큰
# SSH 키 + 토큰 병행 사용
# 환경 변수에 영구 설정
```

### 보조 노트북
```bash
# 제한된 권한 토큰
# .env 파일 사용
# 필요시에만 설정
```

### 서버/클라우드 인스턴스
```bash
# 최소 권한 토큰
# 환경 변수 설정
# 로그 모니터링
```

## 🔧 트러블슈팅

### Q: 토큰이 작동하지 않아요
```bash
# 토큰 유효성 확인
curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user

# GitHub CLI 인증 상태 확인
gh auth status
```

### Q: SSH 키 설정이 어려워요
```bash
# SSH 키 생성
ssh-keygen -t ed25519 -C "your_email@example.com"

# 공개 키 복사
cat ~/.ssh/id_ed25519.pub

# GitHub → Settings → SSH keys에 등록
```

### Q: .env 파일이 Git에 올라갔어요
```bash
# 즉시 파일 제거
git rm --cached .env
git commit -m "Remove .env file"

# 토큰 교체 (보안상 필수)
# GitHub에서 기존 토큰 삭제 후 새 토큰 생성
```

## 📱 모바일/태블릿에서 작업

### GitHub Codespaces 사용
```bash
# 웹 브라우저에서 코드 편집
# 토큰은 Codespaces 환경에만 저장
# 로컬 디바이스에 토큰 저장 불필요
```

### GitHub Mobile App
```bash
# 코드 리뷰 및 간단한 편집
# 별도 토큰 설정 불필요
# OAuth 로그인 사용
```

## 🔄 정기 점검 체크리스트

### 월간 점검
- [ ] 토큰 유효성 확인
- [ ] 권한 사용 내역 검토
- [ ] 불필요한 토큰 제거

### 분기별 점검
- [ ] 토큰 교체 고려
- [ ] SSH 키 상태 확인
- [ ] 보안 로그 검토

### 연간 점검
- [ ] 전체 토큰 갱신
- [ ] 권한 정책 재검토
- [ ] 백업 전략 점검

## 🚨 응급 상황 대응

### 토큰 유출 시
1. **즉시 토큰 비활성화** (GitHub → Settings → Developer settings)
2. **새 토큰 생성** (다른 이름으로)
3. **모든 컴퓨터 토큰 업데이트**
4. **Git 히스토리 확인** (토큰이 커밋에 포함되었는지)
5. **보안 로그 검토**

### 컴퓨터 분실 시
1. **SSH 키 제거** (GitHub → Settings → SSH keys)
2. **관련 토큰 비활성화**
3. **새 SSH 키 생성**
4. **새 토큰 발급**

이 가이드를 따라하면 안전하고 효율적으로 여러 컴퓨터에서 작업할 수 있습니다!
