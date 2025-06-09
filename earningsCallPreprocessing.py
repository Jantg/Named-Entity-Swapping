import os
import io
import re
# used for detecting exec names
import spacy
nlp = spacy.load("en_core_web_trf")

# used for matching company names and exec names
# from the list of participants to do DI suppression
from fuzzywuzzy import fuzz
from cleanco import basename

# packages to read and write pdf, the canvas soultion is not clean
# CHANGE NEEDED IN FUTURE
from pypdf import PdfReader,PdfWriter
from pypdf.errors import PdfStreamError
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

def WriteNewPdf(file,folder):
    os.makedirs(folder, exist_ok=True)
    try:
        reader = PdfReader(file)
    except PdfStreamError:
        None
    outputfile = PdfWriter()
    firstpage = reader.pages[0].extract_text()
    try:
        ticker = re.search(r'(NYSE|NasdaqGS|XTRA|BATS|ASX|NasdaqGM|TSX):([A-Z\.]+)', firstpage).group(2)
    except AttributeError:
        print('Ticker ' + ticker +' not found')
        None
    flag_presstart = False
    flag_edwards = False #it has not Question and Answer header
    writefilename = ticker + '.pdf'
    if writefilename in [x for x in os.listdir(folder)]:
        writefilename = writefilename.split('.')[0] + '_new.pdf'
    for page_id, page in enumerate(reader.pages):
        pagetxt = page.extract_text()
        if bool(re.search(r'Table of Contents', pagetxt)):
            if not bool(re.search(r'Presentation', pagetxt)):
                print('No Presentation found on file:' + str(file))
                break
        elif bool(re.search(r'Call Participants', pagetxt)):
            # extract Exec names
            r = re.compile(r'[Tt][Hh][Ee]\s')
            r2 = re.compile(r'(International)|(Materials)')
            companyname = r.sub('',basename(re.search(r'\n(.*)[\n]*(NYSE|NasdaqGS|XTRA|BATS|ASX|NasdaqGM|TSX)',
                                                          firstpage).group(1)))
                                
            flag_edwards = bool(re.search(r'Edwards',firstpage))

            #str(nlp(pagetxt).ents[0]).split(' FQ')[0] #first entry of the Call participants page is company name
            if (companyname.isupper()) or (len(companyname.split())==1):
                companyname_abbrv = companyname
            else:
                companyname_abbrv = ''.join([i[0] for i in companyname.split()])
            # remove Middle names like Jim T. Johnes -> Jim Johnes
            execnames = [re.sub(r'[A-Z]\.','',str(e.text)) for e in nlp(pagetxt.split('ANALYSTS')[0]).ents if e.label_ =='PERSON']
        else:
            if bool(re.search(r'Presentation', pagetxt)):
                flag_presstart = True
            if flag_presstart:
                # when reaching QA page finish
                if bool(re.search(r'Question and Answer\n', pagetxt)):
                    break
                if flag_edwards:
                    if page_id == 8:
                        break
                doc = nlp(pagetxt)
                for execid, execname in enumerate(execnames):
                    matchednames = sorted(list(set([str(e.text) for e in doc.ents \
                     if (fuzz.partial_ratio(execname.lower(),str(e.text).lower())>=70) and e.label_ =='PERSON'])),
                                          reverse=True)
                    # sorted(,reverse=T) so that we first remove names like Jim Johnes and then remove Jim
                    for matchedname in matchednames:
                        pagetxt = pagetxt.replace(matchedname, '[Executive' + str(execid)+']')
                #for comnameid, comname in enumerate([companyname,companyname_abbrv]):
                matchedcomnames = sorted(list(set([str(e.text).split(' FQ')[0] for e in doc.ents\
                     if fuzz.partial_ratio(companyname.lower(),str(e.text).lower())>=70])),
                                         reverse=True)
                # handling some exceptions e.g., EPS for Pepsi
                if 'EPS' in matchedcomnames:
                    matchedcomnames.remove('EPS')
                #print(matchedcomnames, companyname)
                for matchedcomname in matchedcomnames:
                    pagetxt = pagetxt.replace(matchedcomname,'[Company Name]')
                matchedcomnames_abbrv = set([str(e.text).split(' FQ')[0] for e in doc.ents\
                     if fuzz.partial_ratio(companyname_abbrv,str(e.text))>70])
                #print(matchedcomnames_abbrv, companyname_abbrv)
                for matchedcomnames_abbrv in matchedcomnames_abbrv:
                    pagetxt = pagetxt.replace(matchedcomnames_abbrv,'[Company Name]')
                ent_urls = [str(e.text) for e in doc if e.like_url]
                for ent_url in ent_urls:
                    pagetxt = pagetxt.replace(ent_url,'[URL]')
                packet = io.BytesIO()
                can = canvas.Canvas(packet, pagesize = letter)
                can.drawString(10,100,
                               re.split(r'marketintelligence \d+',
                                       pagetxt)[-1].replace('\n',' '))
                can.save()
                packet.seek(0)
                new_pdf = PdfReader(packet)
                outputfile.add_page(new_pdf.pages[0])
                packet.close()          
        outputfile.write(os.path.join(folder,writefilename))
        
def getMeta(file_name):
     return {"file_name":file_name,"file_path":""}
