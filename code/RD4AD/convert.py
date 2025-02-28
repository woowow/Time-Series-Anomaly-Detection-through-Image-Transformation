from PIL import Image

# filename = "./swat/FIT401_10_100/ground_truth/anomaly/FIT401_10_100_GT_15.png"
# image = Image.open(filename)
# image_gray = image.convert("L")
# image_gray.save("./swat/FIT401_10_100/ground_truth/anomaly/FIT401_10_100_15.png")

# filename = "./temporal_rate/LIT101_50_200_0.png"
# image = Image.open(filename)
# image_gray = image.convert("L")
# image_gray.save("./swat/LIT101_50_200/ground_truth/rate10/LIT101_50_200_0.png")


filename = "./swat/LIT101_50_200/ground_truth/interpolation/LIT101_50_200_0.png"
img = Image.open(filename)
img_gr = img.convert("L")
img_gr.save("./swat/LIT101_50_200/ground_truth/interpolation/LIT101_50_200_0.png")

filename = "./swat/LIT101_50_200/ground_truth/interpolation/LIT101_50_200_11.png"
img = Image.open(filename)
img_gr = img.convert("L")
img_gr.save("./swat/LIT101_50_200/ground_truth/interpolation/LIT101_50_200_11.png")

filename = "./swat/LIT101_50_200/ground_truth/interpolation/LIT101_50_200_28.png"
img = Image.open(filename)
img_gr = img.convert("L")
img_gr.save("./swat/LIT101_50_200/ground_truth/interpolation/LIT101_50_200_28.png")

filename = "./swat/LIT101_50_200/ground_truth/interpolation/LIT101_50_200_36.png"
img = Image.open(filename)
img_gr = img.convert("L")
img_gr.save("./swat/LIT101_50_200/ground_truth/interpolation/LIT101_50_200_36.png")

filename = "./swat/LIT101_50_200/ground_truth/interpolation/LIT101_50_200_38.png"
img = Image.open(filename)
img_gr = img.convert("L")
img_gr.save("./swat/LIT101_50_200/ground_truth/interpolation/LIT101_50_200_38.png")

filename = "./swat/LIT101_50_200/ground_truth/interpolation/LIT101_50_200_39.png"
img = Image.open(filename)
img_gr = img.convert("L")
img_gr.save("./swat/LIT101_50_200/ground_truth/interpolation/LIT101_50_200_39.png")

# img_cr = img_gr.convert("RGB")
# img_cr.save("./c_t/convert_RGB.png")

