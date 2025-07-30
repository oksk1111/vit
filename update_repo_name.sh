#!/bin/bash
# 저장소 이름 변경 후 실행할 스크립트

echo "=== 저장소 이름 변경 후 설정 업데이트 ==="

# 1. 원격 URL 변경
git remote set-url origin https://github.com/oksk1111/vit_test.git
echo "✓ 원격 저장소 URL 업데이트 완료"

# 2. README.md의 clone URL 수정
sed -i 's|git clone https://github.com/oksk1111/vit.git|git clone https://github.com/oksk1111/vit_test.git|g' README.md
sed -i 's|cd vit|cd vit_test|g' README.md
echo "✓ README.md 클론 URL 업데이트 완료"

# 3. 변경사항 커밋 및 푸시
git add README.md
git commit -m "Update README.md: Fix clone URL after repository rename"
git push origin main
echo "✓ 변경사항 GitHub에 푸시 완료"

echo ""
echo "🎉 모든 설정이 완료되었습니다!"
echo "새로운 저장소 URL: https://github.com/oksk1111/vit_test"
