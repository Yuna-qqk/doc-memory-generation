"""
非遗文献数据预处理脚本
将原始 txt/json 文件清洗并转换为 MemoRAG 训练所需格式：{"text": "..."}
"""
import json
import os
import re
import argparse
from pathlib import Path


def clean_text(text: str) -> str:
    """清洗文本：去除 HTML tag、多余空白、特殊字符"""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[a-z]+;', '', text)
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
    return text.strip()


def split_long_text(text: str, max_chars: int = 60000) -> list:
    """
    长文本按段落边界切分，每段不超过 max_chars 字符。
    Qwen2 tokenizer 中文字符约 0.6 token/char，所以 60K char ≈ 36K token。
    """
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len > max_chars and current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            current_chunk = [para]
            current_len = para_len
        else:
            current_chunk.append(para)
            current_len += para_len

    if current_chunk:
        chunks.append('\n\n'.join(current_chunk))

    return chunks


def process_input_files(input_dir: str, output_path: str, max_chars: int = 60000,
                        train_ratio: float = 0.9):
    """
    读取 input_dir 下所有 .txt 和 .json 文件，
    清洗后输出为训练格式的 JSON。
    """
    all_texts = []
    input_path = Path(input_dir)

    txt_files = list(input_path.glob("**/*.txt"))
    json_files = list(input_path.glob("**/*.json"))

    print(f"Found {len(txt_files)} txt files, {len(json_files)} json files")

    # 处理 txt 文件
    for fpath in txt_files:
        try:
            text = fpath.read_text(encoding='utf-8')
            text = clean_text(text)
            if len(text) < 100:
                continue
            chunks = split_long_text(text, max_chars)
            for chunk in chunks:
                if len(chunk) >= 100:
                    all_texts.append({"text": chunk})
        except Exception as e:
            print(f"  Skip {fpath.name}: {e}")

    # 处理 json 文件（假设 {"text": "..."} 或 [{"text": "..."}] 格式）
    for fpath in json_files:
        try:
            data = json.loads(fpath.read_text(encoding='utf-8'))
            if isinstance(data, dict):
                data = [data]
            for item in data:
                raw = item.get("text", "") or item.get("content", "") or item.get("body", "")
                if isinstance(raw, list):
                    raw = "\n".join(raw)
                text = clean_text(str(raw))
                if len(text) < 100:
                    continue
                chunks = split_long_text(text, max_chars)
                for chunk in chunks:
                    if len(chunk) >= 100:
                        all_texts.append({"text": chunk})
        except Exception as e:
            print(f"  Skip {fpath.name}: {e}")

    print(f"Total processed texts: {len(all_texts)}")

    # 划分训练集和验证集
    split_idx = int(len(all_texts) * train_ratio)
    train_data = all_texts[:split_idx]
    eval_data = all_texts[split_idx:]

    # 保存
    train_path = os.path.join(os.path.dirname(output_path), "train.json")
    eval_path = os.path.join(os.path.dirname(output_path), "eval.json")

    json.dump(train_data, open(train_path, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    json.dump(eval_data, open(eval_path, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)

    total_chars = sum(len(d['text']) for d in all_texts)
    print(f"Train: {len(train_data)} samples → {train_path}")
    print(f"Eval:  {len(eval_data)} samples → {eval_path}")
    print(f"Total chars: {total_chars}, est tokens: {total_chars * 0.6:.0f}")


def generate_sample_data(output_dir: str):
    """如果没有真实数据，生成一份示例数据用于测试训练流程"""
    samples = [
        {
            "text": "景德镇手工制瓷技艺是中国陶瓷文化的杰出代表，其历史可追溯至汉代。"
                    "景德镇位于江西省东北部，因宋代景德年间设镇而得名。"
                    "景德镇手工制瓷技艺的核心工序包括选料、粉碎、淘洗、练泥、拉坯、修坯、"
                    "画坯、上釉和烧窑等步骤。其中拉坯是将泥料置于辘轳上，通过手工旋转成型，"
                    "是决定瓷器造型的关键工序。修坯则是在坯体半干时进行修整，使其厚薄均匀、"
                    "表面光滑。上釉工艺分为浸釉、吹釉、荡釉等多种方法，不同器型采用不同上釉方式。"
                    "景德镇瓷器以白如玉、明如镜、薄如纸、声如磬著称于世。"
                    "青花瓷、粉彩瓷、玲珑瓷、颜色釉瓷是景德镇的四大传统名瓷。"
                    "青花瓷以氧化钴为着色剂，在坯体上绘制纹饰后施透明釉，经高温一次烧成，"
                    "呈色稳定、发色青翠。粉彩瓷则是在烧成的白瓷上用含砷的玻璃白打底，"
                    "再将颜料施于其上，用笔洗开，使颜色产生浓淡变化，经低温焙烧而成。"
                    "景德镇手工制瓷技艺于2006年被列入第一批国家级非物质文化遗产名录。"
                    "代表性传承人包括王锡良、张松茂、秦锡麟等工艺美术大师。"
        },
        {
            "text": "苏绣是苏州地区传统刺绣技艺的统称，与湘绣、粤绣、蜀绣并称中国四大名绣。"
                    "苏绣的历史可追溯到春秋时期，到了明清两代达到鼎盛。"
                    "苏绣的主要针法包括齐针、散套、施针、虚实针、乱针、滚针、切针、"
                    "平金、打点、戳纱、接针等数十种。其中散套针法是最常用的针法之一，"
                    "通过分层刺绣使色彩渐变过渡自然。乱针绣是二十世纪三十年代由杨守玉创制的，"
                    "打破了传统刺绣排比其针、密接其线的规律，以交叉的线条表现物象。"
                    "苏绣的艺术特色可概括为'平、齐、细、密、匀、顺、和、光'八个字。"
                    "苏州地区形成了以沈寿、杨守玉、任嘒閒等为代表的刺绣艺术流派。"
                    "苏绣的题材广泛，包括花鸟、山水、人物、走兽等。"
                    "双面绣是苏绣的绝技之一，在同一块底料上绣出正反两面图像，"
                    "两面图案同样精致，甚至可以是不同的图案。"
                    "苏绣于2006年被列入第一批国家级非物质文化遗产名录。"
        },
        {
            "text": "雕漆技艺是中国传统漆器工艺的重要门类，其制作过程是在涂有数十层甚至"
                    "数百层漆的胎体上雕刻纹饰。雕漆工艺始于唐代，盛于明清。"
                    "雕漆的主要工序包括制胎、涂漆、描图、雕刻、磨光等。"
                    "制胎是在胎体上反复涂刷大漆，每涂一层需等待完全干燥后再涂下一层，"
                    "涂到所需厚度后才能进行雕刻。雕刻是雕漆工艺的核心环节，"
                    "包括刺、起、片、铲、勾等多种刀法。"
                    "北京雕漆和扬州雕漆是最具代表性的两大流派。"
                    "北京雕漆以朱红为主色调，刀法刚劲有力，纹饰以花鸟、人物为主。"
                    "扬州雕漆则以剔红、剔犀最为著名，色彩丰富，刀法精细圆润。"
                    "剔犀是雕漆的一种特殊形式，是用两种或三种色漆在胎体上有规律地交替涂刷，"
                    "达到一定厚度后雕刻出云纹、回纹等图案，刀口断面呈现出不同颜色的漆层纹路。"
                    "雕漆技艺于2006年被列入第一批国家级非物质文化遗产名录。"
                    "代表性传承人包括文乾刚、殷秀云等。"
        },
        {
            "text": "宜兴紫砂陶制作技艺是江苏省宜兴市丁蜀镇特有的传统手工制陶技艺。"
                    "宜兴紫砂陶始于北宋，盛于明清，以独特的紫砂泥料和精湛的手工成型工艺闻名。"
                    "紫砂泥料主要有紫泥、红泥、绿泥三种基本类型，经过开采、风化、粉碎、"
                    "过筛、练泥等工序制成可塑泥料。成型工艺以手工拍打成型为主，"
                    "先将泥料拍打成片状，再围成筒状，然后通过拍打、镶接逐步成型。"
                    "紫砂壶的制作工具有搭子、拍子、矩车、尖刀、明针等数十种。"
                    "明针是紫砂壶表面处理的重要工具，用于修整壶身表面使其光滑。"
                    "紫砂陶的烧成温度一般在1150-1200摄氏度之间，采用氧化焰或还原焰烧成。"
                    "历代紫砂名家包括供春、时大彬、陈鸣远、邵大亨、顾景舟等。"
                    "供春被誉为紫砂壶的始祖，其所制树瘿壶仿照银杏树瘿的形态，古朴自然。"
                    "顾景舟是二十世纪最著名的紫砂艺人，其作品造型端庄、线条流畅。"
                    "宜兴紫砂陶制作技艺于2006年被列入第一批国家级非物质文化遗产名录。"
        },
        {
            "text": "京剧是中国最具代表性的戏曲剧种，形成于清代乾隆年间（十八世纪末），"
                    "以徽剧和汉剧为基础，吸收昆曲、秦腔等剧种的特点逐步融合发展而成。"
                    "京剧的表演艺术包括唱、念、做、打四种基本表现手段。"
                    "唱指声腔演唱，京剧的唱腔以西皮、二黄为主要声腔体系。"
                    "西皮曲调活泼刚劲，适合表现欢快激昂的情绪；二黄曲调深沉稳重，适合表现悲壮哀婉的情绪。"
                    "念指念白，包括韵白和京白两种，韵白富于音乐性，京白接近北京话。"
                    "做指身段表演，包括手势、眼神、身段、步法等程式化动作。"
                    "打指武打表演，包括把子功和毯子功。"
                    "京剧的角色分为生、旦、净、丑四大行当。生行包括老生、小生、武生等；"
                    "旦行包括青衣、花旦、刀马旦、老旦等；净行又称花脸，以面部勾画脸谱为特征；"
                    "丑行以鼻梁上涂白粉为标志，分为文丑和武丑。"
                    "京剧脸谱是净行角色特有的面部化妆艺术，通过色彩和图案表现人物性格。"
                    "红色代表忠勇、白色代表奸诈、黑色代表刚直、蓝色代表勇猛。"
                    "梅兰芳、程砚秋、尚小云、荀慧生被誉为京剧四大名旦。"
                    "梅兰芳创立了梅派艺术，其表演风格雍容华贵、典雅大方。"
                    "京剧于2010年被列入联合国教科文组织人类非物质文化遗产代表作名录。"
        },
        {
            "text": "龙泉青瓷烧制技艺是浙江省龙泉市传承千年的传统制瓷技艺。"
                    "龙泉窑始烧于三国两晋时期，至南宋达到鼎盛，是中国陶瓷史上延续时间最长的窑系。"
                    "龙泉青瓷以粉青和梅子青釉色最为著名。粉青釉色青中带蓝，如青玉般温润；"
                    "梅子青釉色青翠欲滴，如青梅般鲜嫩。"
                    "龙泉青瓷的制瓷原料主要是当地出产的瓷土和紫金土，釉料以石灰碱釉为主。"
                    "石灰碱釉含有较高的钾钠含量，在高温下黏度大，不易流淌，可以施厚釉，"
                    "烧成后釉层肥厚滋润，具有玉质感。"
                    "龙泉青瓷的成型方法包括拉坯、模制、泥片贴筑等。"
                    "装饰技法主要有刻花、划花、印花、贴花和浮雕等，以刻花最为常见。"
                    "烧成采用龙窑或馒头窑，以松柴为燃料，在还原气氛下烧成。"
                    "南宋时期龙泉窑形成了以章生一、章生二兄弟为代表的两大流派，"
                    "章生一所制瓷器称为'哥窑'，以釉面开片纹为特征；"
                    "章生二所制瓷器称为'弟窑'，以釉色青润无纹为特征。"
                    "龙泉青瓷烧制技艺于2006年被列入第一批国家级非物质文化遗产名录。"
                    "2009年，龙泉青瓷传统烧制技艺被列入联合国教科文组织人类非物质文化遗产代表作名录。"
        },
    ]

    train_path = os.path.join(output_dir, "train.json")
    eval_path = os.path.join(output_dir, "eval.json")

    json.dump(samples[:5], open(train_path, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    json.dump(samples[5:], open(eval_path, 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)

    print(f"Generated {len(samples[:5])} train + {len(samples[5:])} eval samples")
    total_chars = sum(len(d['text']) for d in samples)
    print(f"Total chars: {total_chars}, est tokens: {total_chars * 0.6:.0f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ICH training data")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Directory containing raw .txt/.json files")
    parser.add_argument("--output_dir", type=str, default="data/ich",
                        help="Output directory for train.json and eval.json")
    parser.add_argument("--sample", action="store_true", default=True,
                        help="Generate sample data if no input_dir provided")
    args = parser.parse_args()

    if args.input_dir and os.path.isdir(args.input_dir):
        process_input_files(args.input_dir, os.path.join(args.output_dir, "train.json"))
    else:
        print("No input_dir, generating sample ICH data...")
        generate_sample_data(args.output_dir)
