import PyPDF2
import fitz

class PdfReader:
    def __init__(self, filename):
        self.filename = filename

    def total_pages(self):
        with open(self.filename, 'rb') as f:
            pdf_reader = PyPDF2.PdfFileReader(f)
            return pdf_reader.numPages 

    def read(self):
        with open(self.filename, 'rb') as f:
            pdf_reader = PyPDF2.PdfFileReader(f)
            num_pages = pdf_reader.numPages
            count = 0
            text = ''
            while count < num_pages:
                text += pdf_reader.getPage(count).extractText()
                count += 1
            return text

    def read_pages(self, start_page, end_page):
        with open(self.filename, 'rb') as f:
            pdf_reader = PyPDF2.PdfFileReader(f)
            text = ''
            for page in range(start_page, end_page):
                text += pdf_reader.getPage(page).extractText()
            return text

    def extract_images(self):
        doc = fitz.open(self.filename)
        for page_index in range(len(doc)):
            for img in doc.get_page_images(page_index):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n < 5:       # GRAY or RGB
                    pix.save(f"{xref}.png")
                else:               # convert to RGB 
                    pix1 = fitz.Pixmap(fitz.csRGB, pix)
                    pix1.save(f"{xref}.png")
                    pix1 = None
                pix = None

class ExtractedText(PdfReader):
    def __init__(self, filename, output_filename):
        super().__init__(filename)
        self.output_filename = output_filename

    def save(self,start_page, end_page):
        with open(self.filename,'rb') as f:
            pdf_reader = PyPDF2.PdfFileReader(f)
            text = ''
            for page in range(start_page, end_page):
                text += pdf_reader.getPage(page).extractText()
            with open(self.output_filename, 'w',encoding='utf-8') as f:
                f.write(text)
