import pandas as pd
import pdfplumber

pdf = pdfplumber.open("test.pdf")
# print(pdf.metadata)
# print(pdf.pages)

pages = pdf.pages
# print(pages[0].width, pages[0].height)
# pages[0].to_image().show()
# pages[1].to_image().show()

p1_text = pages[0].extract_text()
# p1_text = pages[0].extract_text(layout=True)
# print(p1_text)

p1_table = pages[0].extract_table()
# print(p1_table)
df = pd.DataFrame(p1_table[1:], columns=p1_table[0])

p1_tables = pages[0].extract_tables()
# print(len(p1_tables))

p1_debug_table = pages[0].debug_tablefinder()
# print(type(p1_debug_table))
# print(p1_debug_table.tables)

# print(pages[1].images)

img = pages[1].images[0]
bbox = (img["x0"], img["top"], img["x1"], img["bottom"])
cropped_page = pages[1].crop(bbox)
# cropped_page.to_image().show()
# cropped_page.to_image(antialias=True).show()
# cropped_page.to_image(resolution=1080).show()
# im = cropped_page.to_image(antialias=True, resolution=1080)
# im.save("pdf_image_test.png")



