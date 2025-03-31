import csv

# 향미성분 csv 파일에서 compound_name이 존재하는 행의 인덱스
COMPOUND_NAME = 3

input_file = 'dataset/input_profile_v250319.csv'
node_file = 'dataset/nodes_191120.csv'
edge_file = 'dataset/edges_191120.csv'

def add_to_nodes(id, name, node_type):
    with open(node_file, 'a', encoding='utf-8', newline='') as node_f:
        node_writer = csv.writer(node_f)
        # 노드의 유형에 따라 네번째 인자를 지정
        if node_type == 'compound':  
            fourth_arg = 'food'
        elif node_type == 'ingredient':
            fourth_arg = 'no_hub'
        else:
            fourth_arg = ''

        new_node = [id, name, '', node_type, fourth_arg]
        node_writer.writerow(new_node)  # 노드 CSV 파일에 추가
        print("node added: ", new_node)
        return new_node

def find_from_nodes(name, node_type):
    with open(node_file, 'r', encoding='utf-8') as node_f:
        node_contents = csv.reader(node_f)
        # 헤더 건너뛰기
        next(node_contents, None)
        id_count = 0
        for row in node_contents:
            # 빈 행 또는 불완전한 행 무시
            if len(row) < 2:
                continue
            if row[1] == name:
                print("found node: ", row)
                return row  # 기존 노드 반환
            id_count += 1
        # 존재하지 않으면 새 노드 추가
        new_node = add_to_nodes(id_count, name, node_type)
        return new_node

def find_node_type(node):
    if node[3] == 'ingredient':
        return 'ingr'
    elif node[3] == 'compound':
        if node[4] == 'food':
            return 'fcomp'
        elif node[4] == 'drug':
            return 'dcomp'
    return None

def add_to_edges(node_1, node_2):
    with open(edge_file, 'a', encoding='utf-8', newline='') as edge_f:
        edge_writer = csv.writer(edge_f)
        node_1_type = find_node_type(node_1)
        node_2_type = find_node_type(node_2)
        edge_type = ''

        # 엣지의 방향과 관계없이 'ingredient'가 앞에 오도록 구성
        if node_1_type == 'ingr':
            edge_type = f"{node_1_type}-{node_2_type}"
        elif node_2_type == 'ingr':
            edge_type = f"{node_2_type}-{node_1_type}"

        new_edge = [node_1[0], node_2[0], '', edge_type]
        edge_writer.writerow(new_edge)
        print("edge added: ", new_edge)
        return new_edge

def find_from_edges(node_1, node_2):
    with open(edge_file, 'r', encoding='utf-8') as edge_f:
        edge_contents = csv.reader(edge_f)
        # 헤더 건너뛰기
        next(edge_contents, None)
        for row in edge_contents:
            # 빈 행 또는 불완전한 행 무시
            if len(row) < 2:
                continue
            # 노드의 ID(첫번째 요소)를 기준으로 비교
            if (row[0] == node_1[0] and row[1] == node_2[0]) or (row[0] == node_2[0] and row[1] == node_1[0]):
                print("found edge: ", row)
                return row
        # 존재하지 않으면 새 엣지 추가
        new_edge = add_to_edges(node_1, node_2)
        return new_edge

if __name__ == '__main__':
    with open(input_file, 'r', encoding='utf-8') as input_f:
        input_contents = csv.reader(input_f)
        # 술의 이름을 저장할 리스트
        alcohol_name_list = []
        # 상위 3줄은 건너뜀 (헤더 등)
        for i in range(3):
            next(input_contents, None)

        first_flag = True  # 첫 번째 데이터 행에서 술 이름을 저장하기 위한 플래그
        for row in input_contents:
            if first_flag:
                # 첫 번째 행에 있는 각 셀의 값에서 첫 공백 전까지의 문자열을 alcohol_name_list에 저장
                for name in row:
                    slice_index = name.find(' ')
                    if slice_index != -1:
                        alcohol_name_list.append(name[:slice_index])
                    else:
                        alcohol_name_list.append(name)
                first_flag = False
                continue

            # compound 이름 처리: 공백 제거, 언더바 치환, 소문자 변환
            compound_name = row[COMPOUND_NAME].strip().replace(' ', '_').lower()
            node_1 = find_from_nodes(compound_name, 'compound')

            # 4번째 열부터 마지막 열까지를 농도 리스트로 취급
            concentration_list = row[4:]
            for i in range(len(concentration_list)):
                if concentration_list[i] != '':
                    # alcohol_name_list에서 해당 인덱스의 이름을 가져옴
                    ingredient_name = alcohol_name_list[i]
                    node_2 = find_from_nodes(ingredient_name, 'ingredient')
                    find_from_edges(node_1, node_2)
