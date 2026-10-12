"""Restore native print/view settings omitted by Artifact Tool's XLSX exporter."""
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as ET

root = Path(__file__).resolve().parent.parent
target = root / 'MDA_WBS_20261006-20261106.xlsx'
ns = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with ZipFile(root / 'MDA_WBS_수정본.xlsx') as source, ZipFile(target) as current:
    parts = {name: current.read(name) for name in current.namelist()}
    a = ET.fromstring(source.read('xl/worksheets/sheet1.xml'))
    b = ET.fromstring(parts['xl/worksheets/sheet1.xml'])
    for tag in ('sheetViews', 'pageMargins', 'pageSetup'):
        orig, edited = a.find('s:' + tag, ns), b.find('s:' + tag, ns)
        if orig is not None:
            if edited is not None:
                b.replace(edited, deepcopy(orig))
            else:
                b.append(deepcopy(orig))
    for name in ('xl/worksheets/_rels/sheet1.xml.rels', 'xl/printerSettings/printerSettings1.bin'):
        parts[name] = source.read(name)
    parts['xl/worksheets/sheet1.xml'] = ET.tostring(b, xml_declaration=True, encoding='utf-8')
    book = ET.fromstring(parts['xl/workbook.xml'])
    calc = book.find('s:calcPr', ns)
    if calc is None:
        calc = ET.SubElement(book, '{' + ns['s'] + '}calcPr')
    calc.set('calcMode', 'auto')
    calc.set('fullCalcOnLoad', '1')
    calc.set('forceFullCalc', '1')
    parts['xl/workbook.xml'] = ET.tostring(book, xml_declaration=True, encoding='utf-8')
    ct = ET.fromstring(parts['[Content_Types].xml'])
    srcct = ET.fromstring(source.read('[Content_Types].xml'))
    for entry in srcct:
        if entry.get('Extension') == 'bin' or 'printerSettings' in entry.get('PartName', ''):
            if not any(dict(x.attrib) == dict(entry.attrib) for x in ct):
                ct.append(deepcopy(entry))
    parts['[Content_Types].xml'] = ET.tostring(ct, xml_declaration=True, encoding='utf-8')
temporary = target.with_suffix('.tmp')
with ZipFile(temporary, 'w', ZIP_DEFLATED) as out:
    for name, data in parts.items():
        out.writestr(name, data)
temporary.replace(target)
print('Native print/view settings preserved; automatic calculation enabled.')
