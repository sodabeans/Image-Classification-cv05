import os
import shutil

list_gender_male = [
    "000667",
    "000725",
    "000736",
    "000817",
    "003780",
    "006504",
    "000225",
    "000664",
    "000767",
    "001509",
    "003113",
    "003223",
    "004281",
    "006359",
    "006360",
    "006361",
    "006362",
    "006363",
    "006364",
    "006424",
]
list_gender_female = [
    "001498-1",
    "004432",
    "005223",
]
list_age_29 = ["001009", "001064", "001637", "001666", "001852"]
list_age_60 = ["004348"]

# incorrect와 normal 바꾸는 거는 직접하셔야합니다!


def modify_filename(path, m_list, m_col, m_value: str):
    for p in os.listdir(path):
        for i in m_list:
            if p.startswith(i):
                p_list = p.split("_")
                p_list[m_col] = m_value
                m_p = ("_").join(p_list)
                # print(m_p)
                # print(path+'/'+p)
                # print(path+'/'+m_p)
                shutil.move(os.path.join(path, p), os.path.join(path, m_p))


if __name__ == "__main__":
    modify_filename(path, list_age_60, 3, "61")
    modify_filename(path, list_age_29, 3, "28")
    modify_filename(path, list_gender_female, 1, "female")
    modify_filename(path, list_gender_male, 1, "male")

