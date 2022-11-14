import os
from PyPDF2 import PdfMerger
from datetime import date

directory = r'\\homedir-cifs.nyumc.org\bicen01\Personal\lafayette, esther/'
merge_files = os.listdir(directory)
datestr = date.today().strftime('%b-%d-%Y')
merger = PdfMerger()
for pdf in merge_files:
    merger.append(directory + pdf)
merger.write(directory + datestr + '.pdf')