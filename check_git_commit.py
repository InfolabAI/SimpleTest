import os
import subprocess


def save_git_commit_number(path, output_file):
    if not os.path.isdir(path):
        os.makedirs(path)

    # Git 커밋 번호 가져오기
    commit_number = (
        subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    )

    # 커밋 번호를 파일에 저장
    with open(path + output_file, "w") as file:
        file.write(commit_number)

    # 사용 예시
    # output_file = "commit_number.txt"
    # save_git_commit_number(output_file)


def check_git_status(path, output_file):
    # Git 상태 확인
    status_output = (
        subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
    )

    # 변경된 파일이나 스테이징되지 않은 파일이 있는 경우 경고 메시지 출력 후 종료 선택. 없으면 commit 정보 저장
    is_commited = True
    modified_list = []
    for st in status_output.split("\n"):
        if st.strip()[0] == "M" and "hyper.json" not in st.strip():
            is_commited = False
            modified_list.append(st.strip())

    if not is_commited:
        print(f"Warning: There are uncommitted changes in the Git repository.")
        for st in modified_list:
            print(st)
        user_input = input("Please enter 'n' to terminate: ")
        if user_input == "n":
            exit()
    else:
        save_git_commit_number(path, output_file)
