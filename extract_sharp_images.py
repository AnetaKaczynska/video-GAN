import argparse
import multiprocessing
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

import xmltodict
from moviepy.editor import VideoFileClip
from unidecode import unidecode


def get_xml(xml_path):
    dfs = []
    with open(xml_path) as file:
        f = file.read()
        xml = xmltodict.parse(f)
        if xml["annotation"] is None:
            return []

        if not isinstance(xml["annotation"]["frame"], list):
            j = [xml["annotation"]["frame"]]
        else:
            j = xml["annotation"]["frame"]

        for i in j:
            number = i["@number"]
            item = i["item"]
            item["@number"] = number
            if item["@name"] != "stopklatka":
                dfs.append(pd.DataFrame(item, index=[0]))

    return dfs


def select_image(image, blur_scale=100):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm >= blur_scale


def processOneFilm(video_path, save_path):
    dir_name = video_path.parent
    df = pd.DataFrame(columns=["name", "tag"])
    if not Path(f"{save_path}/{video_path.stem}_lables.csv").is_file():
        try:
            print(video_path)
            xml_path = dir_name / (video_path.stem + ".xml")

            dd = get_xml(xml_path)
            dd = pd.concat(dd).reset_index(drop=True)
            fns = [(r["@name"], r["@number"]) for _, r in dd.iterrows()]

            video = VideoFileClip(str(video_path))
            fps = video.fps
            duration = video.duration

            # if there are some tags, then we want to start from place where the first tag is
            beg = int(duration // 2) if len(fns) == 0 else int(fns[0][1]) // fps
            end = video.end
            with TemporaryDirectory(
                prefix=f"{video_path.stem}-", dir=save_path
            ) as tempdir:
                video_name = Path(tempdir) / video_path.name
                clip = video.subclip(beg, end)
                params = "crop=1350:576:550:253, scale=256:256"
                clip.write_videofile(
                    str(video_name), ffmpeg_params=["-vf", str(params)]
                )
                clip.close()
                video.close()

                tags = {}
                for tag, num in fns:
                    if beg * fps <= int(num) < end * fps:
                        tags[f"{int(num) - int(beg * fps)}"] = tag

                video = cv2.VideoCapture(str(video_name))

                good_quality = 100
                previous_img = None
                previous_tag = None

                fc = 1
                while (data := video.read()) and data[0]:
                    image = data[1]

                    check_quality = select_image(image, good_quality)
                    img_path = Path(
                        str(save_path / f"{video_path.stem}_{int(fc + beg * fps)}.jpg")
                    )
                    tag = (
                        unidecode("_".join(tags[str(fc)].split()))
                        if str(fc) in tags
                        else None
                    )

                    if tag is not None:
                        cv2.imwrite(str(img_path), image)
                        df = pd.concat(
                            [
                                df,
                                pd.DataFrame.from_dict(
                                    {"name": [img_path.stem], "tag": [tag]}
                                ),
                            ],
                            ignore_index=True,
                        )
                        previous_tag = tag
                        previous_img = image.copy()
                    else:
                        if previous_img is not None and check_quality:
                            first_image_hist = cv2.calcHist(
                                [image], [0], None, [256], [0, 256]
                            )
                            second_image_hist = cv2.calcHist(
                                [previous_img], [0], None, [256], [0, 256]
                            )

                            img_hist_diff = cv2.compareHist(
                                first_image_hist,
                                second_image_hist,
                                cv2.HISTCMP_BHATTACHARYYA,
                            )
                            img_template_probability_match = cv2.matchTemplate(
                                first_image_hist,
                                second_image_hist,
                                cv2.TM_CCOEFF_NORMED,
                            )[0][0]
                            img_template_diff = 1 - img_template_probability_match

                            # taking only 10% of histogram diff, since it's less accurate than template method
                            commutative_image_diff = (
                                img_hist_diff / 10
                            ) + img_template_diff

                            if commutative_image_diff < 0.2:
                                check_quality = False
                            elif commutative_image_diff < 0.4:
                                tag = previous_tag

                        if check_quality:
                            cv2.imwrite(str(img_path), image)
                            df = pd.concat(
                                [
                                    df,
                                    pd.DataFrame.from_dict(
                                        {"name": [img_path.stem], "tag": [tag]}
                                    ),
                                ],
                                ignore_index=True,
                            )
                            previous_tag = tag
                            previous_img = image.copy()

                    fc += 1

                video.release()

            if not df.empty:
                df.to_csv(f"{save_path}/{video_path.stem}_lables.csv", index=False)
                # split images to different folders
                for tag in df["tag"].unique():
                    Path(f"{save_path}/{tag}").mkdir(exist_ok=True, parents=True)

                for index, row in df.iterrows():
                    Path(f"{save_path}/{row['name']}.jpg").rename(
                        f"{save_path}/{row['tag']}/{row['name']}.jpg"
                    )
        except Exception as e:
            print("Corrupted video: ", video_path, e)


def main(opts):
    save_path = Path(opts.sharp_images)
    save_path.mkdir(exist_ok=True, parents=True)

    try:
        num_cores = multiprocessing.cpu_count() // 2
        Parallel(n_jobs=num_cores)(
            delayed(processOneFilm)(
                video_path,
                save_path,
            )
            for video_path in Path(args.raw_films).glob("*.mp4")
        )

    except KeyboardInterrupt:
        print("\033[0;33m\nStop processing film\033[0m")

    # iii = 0
    # for video_path in Path(opts.raw_films).glob("*.mp4"):
    #     processOneFilm(
    #         video_path,
    #         save_path,
    #     )
    #     if iii == 1:
    #         break
    #     iii += 1


def test(opts):
    save_path = Path(opts.sharp_images)

    dfs = None
    files = list(Path(save_path).glob("*.csv"))
    for csv in tqdm(files, total=len(files)):
        df = pd.read_csv(csv)
        df = df.replace({np.nan: None})
        for index, row in tqdm(df.iterrows(), total=len(df), leave=False):
            assert Path(
                f"{save_path}/{row['tag']}/{row['name']}.jpg"
            ).is_file(), (
                f"File '{save_path}/{row['tag']}/{row['name']}.jpg' does not exist!"
            )
        if dfs is None:
            dfs = df
        else:
            dfs = pd.concat([dfs, df]).reset_index(drop=True)
        # Path(csv).unlink()
    print(dfs)
    if not dfs.empty:
        dfs.to_csv(f"{save_path}/lables.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--raw_films", help="Path of directory with raw films", default="/data/Innomed"
    )
    parser.add_argument(
        "--sharp_images",
        help="Path of directory where to save images",
        default="/results/processed_films",
    )
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    print(args)

    if args.test:
        test(args)
    else:
        main(args)
