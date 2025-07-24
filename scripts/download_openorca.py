from datasets import load_dataset

print("📥 Downloading OpenOrca...")
ds = load_dataset("Open-Orca/OpenOrca", split="train")

print("💾 Saving to disk...")
ds.save_to_disk("local_datasets/openorca")

print("✅ Done.")
