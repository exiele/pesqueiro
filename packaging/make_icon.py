import sys

from PIL import Image, ImageDraw

SIZE = 256


def main(target: str) -> None:
    image = Image.new("RGBA", (SIZE, SIZE), (40, 42, 54, 255))
    draw = ImageDraw.Draw(image)
    draw.ellipse((28, 28, SIZE - 28, SIZE - 28), outline=(189, 147, 249, 255), width=10)
    draw.line((SIZE // 2, 52, SIZE // 2, 150), fill=(248, 248, 242, 255), width=6)
    draw.ellipse((SIZE // 2 - 26, 150, SIZE // 2 + 26, 202), fill=(255, 85, 85, 255))
    image.save(target)


main(sys.argv[1])
