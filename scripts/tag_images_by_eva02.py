import argparse
import csv
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import timm
from huggingface_hub import hf_hub_download

def prepare_tagger_files(repo_id: str, save_dir: str):
    """허깅페이스 공식 LFS 호환 다운로더 (CDN 403 에러 방지)"""
    os.makedirs(save_dir, exist_ok=True)
    files = ["model.safetensors", "config.json", "selected_tags.csv"]

    for fname in files:
        dest_path = os.path.join(save_dir, fname)
        if not os.path.exists(dest_path):
            print(f"⬇️ [HF Hub 안전 다운로드] {fname}")
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=fname,
                    local_dir=save_dir,
                    force_download=False
                )
            except Exception as e:
                print(f"⚠️ 다운로드 중 에러 발생 ({fname}): {e}")
        else:
            print(f"✅ 파일 존재함: {fname}")

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            img = Image.open(path)
            # 투명 배경(RGBA)을 흰색 배경 RGB로 정제
            if img.mode in ("RGBA", "LA") or "transparency" in img.info:
                bg = Image.new("RGB", img.size, (255, 255, 255))
                bg.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                img = bg
            else:
                img = img.convert("RGB")

            # 비율 보존을 위한 정사각형 레터박스 패딩 (White Padding)
            w, h = img.size
            max_dim = max(w, h)
            pad_img = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
            pad_img.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))

            tensor = self.transform(pad_img)
            return tensor, str(path)
        except Exception as e:
            return None, str(path)

def collate_fn(batch):
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None, []
    tensors, paths = zip(*batch)
    return torch.stack(tensors), list(paths)

def main():
    parser = argparse.ArgumentParser(description="PyTorch Native WD EVA02 Tagger")
    parser.add_argument("train_data_dir", type=str, help="이미지 폴더 경로")
    parser.add_argument("--repo_id", type=str, default="SmilingWolf/wd-eva02-large-tagger-v3")
    parser.add_argument("--model_dir", type=str, default="models/taggers")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--general_threshold", type=float, default=0.35)
    parser.add_argument("--character_threshold", type=float, default=0.35)
    parser.add_argument("--remove_underscore", action="store_true", help="언더바(_)를 공백으로 변경")
    parser.add_argument("--caption_extension", type=str, default=".txt")
    parser.add_argument("--undesired_tags", type=str, default="")
    args = parser.parse_args()

    # 모델 저장 폴더 준비 및 다운로드
    save_dir = os.path.abspath(os.path.join(args.model_dir, args.repo_id.replace("/", "_")))
    print(f"📦 [모델 확인 및 다운로드] {args.repo_id} -> {save_dir}")
    prepare_tagger_files(args.repo_id, save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 PyTorch GPU 엔진으로 모델 로드 중...")

    # timm 모델 로드 (HF Repo ID지정 + 이미 다운로드된 로컬 checkpoint 사용)
    model = timm.create_model(
        f"hf_hub:{args.repo_id}",
        pretrained=True,
        checkpoint_path=os.path.join(save_dir, "model.safetensors")
    )
    model = model.to(device).eval()

    # 모델 전용 변환(Transform) 해상도(448x448) 및 정규화 자동 적용
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    # selected_tags.csv 읽기
    csv_path = os.path.join(save_dir, "selected_tags.csv")
    tags_info = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # 헤더 스킵 (tag_id, name, category, count)
        for row in reader:
            name, cat = row[1], row[2]
            if args.remove_underscore:
                name = name.replace("_", " ") if len(name) > 3 else name
            tags_info.append((name, cat))

    undesired = set(x.strip() for x in args.undesired_tags.split(",") if x.strip())

    # 이미지 목록 수집
    exts = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    image_paths = [
        p for p in Path(args.train_data_dir).rglob("*")
        if p.suffix.lower() in exts
    ]
    print(f"📸 총 {len(image_paths)}장 이미지 발견. 태깅을 시작합니다.")

    dataset = ImageDataset(image_paths, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

    for tensors, paths in tqdm(loader, desc="EVA02 Tagging"):
        if tensors is None:
            continue
        tensors = tensors.to(device)
        with torch.no_grad():
            outputs = model(tensors)
            probs = torch.sigmoid(outputs).cpu().numpy()

        for path_str, prob in zip(paths, probs):
            char_tags = []
            gen_tags = []

            for i, p in enumerate(prob):
                tag_name, cat = tags_info[i]
                if tag_name in undesired:
                    continue

                if cat == "4" and p >= args.character_threshold:
                    char_tags.append((tag_name, p))
                elif cat == "0" and p >= args.general_threshold:
                    gen_tags.append((tag_name, p))

            # 확률 높은 순 정렬 (캐릭터 태그 우선 배열)
            char_tags.sort(key=lambda x: x[1], reverse=True)
            gen_tags.sort(key=lambda x: x[1], reverse=True)
            
            final_tags = [t[0] for t in char_tags] + [t[0] for t in gen_tags]
            tag_str = ", ".join(final_tags)

            caption_file = os.path.splitext(path_str)[0] + args.caption_extension
            with open(caption_file, "w", encoding="utf-8") as f:
                f.write(tag_str + "\n")

    print("✨ EVA02 태깅 완료!")

if __name__ == "__main__":
    main()
