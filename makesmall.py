import csv

input_path = 'dataset/nodes_191120.csv'
output_path = 'dataset/nodes_191120_lower.csv'  # 원본 보존용

with open(input_path, 'r', encoding='utf-8') as infile, \
     open(output_path, 'w', encoding='utf-8', newline='') as outfile:

    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    for row in reader:
        if len(row) < 2:
            writer.writerow(row)
            continue
        # 두 번째 컬럼(노드 이름)을 소문자로 변환
        row[1] = row[1].lower()
        writer.writerow(row)

print(f"✅ 변환 완료: {output_path}")
