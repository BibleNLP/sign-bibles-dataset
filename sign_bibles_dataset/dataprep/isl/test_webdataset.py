import webdataset as wds

# dataset = wds.WebDataset("../../../test_webdataset/chunk_{001..002}.tar").decode()

dataset = wds.WebDataset(
    "https://huggingface.co/datasets/bridgeconn/sign-bible/resolve/main/chunk_{001..005}.tar"
).decode()

# buffer_size = 1000
# dataset = (
#     wds.WebDataset("https://huggingface.co/datasets/bridgeconn/sign-bible/resolve/main/chunk_001.tar", shardshuffle=False)
#     .shuffle(buffer_size)
#     .decode()
# )

for sample in dataset:
    # print(sample.keys())  # should include 'mp4', 'pose.mp4', 'mask.mp4', 'json'
    mp4_data = sample["mp4"]  # main video
    pose_data = sample["pose.mp4"]  # pose video
    mask_data = sample["mask.mp4"]  # mask video
    json_data = sample["json"]  # JSON metadata

    print(json_data["filename"])
