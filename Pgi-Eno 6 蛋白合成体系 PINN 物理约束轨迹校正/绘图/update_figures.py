"""Update saved figure SVGs without replotting data or recomputing layout.

Keep the original native panel artwork. Assemble it at the positions of the
user-supplied final composites, which omit the source script's panel titles.
All final text is outlined with its original font to prevent font substitution.
"""
from pathlib import Path
import copy
import json
import re
import xml.etree.ElementTree as E

OUT = Path(__file__).resolve().parent
SRC = OUT / '绘图模板'
NS = 'http://www.w3.org/2000/svg'
XL = 'http://www.w3.org/1999/xlink'
E.register_namespace('', NS)
E.register_namespace('xlink', XL)
def tag(s): return '{'+NS+'}'+s
def parse(path):
    return E.parse(path, parser=E.XMLParser(target=E.TreeBuilder(insert_comments=True)))
def by_id(root, id): return next(n for n in root.iter() if n.get('id') == id)
def comments(node): return ' '.join(n.text or '' for n in node.iter() if n.tag is E.Comment)
def style(s): return dict((k.strip(),v.strip()) for k,v in (x.split(':',1) for x in s.split(';') if ':' in x))

# Cached vectors were extracted using the exact PingFang HK collection/face
# confirmed on 20261030. This keeps the original paths, spacing and widths,
# without requiring a Mac font installation or an external working directory.
vector_cache = json.loads((OUT/'字体轮廓缓存.json').read_text(encoding='utf-8'))

def outline_piece(text, attrs, inherited=None):
    css = dict(inherited or {}); css.update(style(attrs.get('style','')))
    size = float(css.get('font-size','12px').replace('px',''))
    family = css.get('font-family', "'Hiragino Sans GB'").strip("'\"")
    x=float(attrs.get('x',0)); y=float(attrs.get('y',0))
    key=json.dumps([size,family,css.get('font-style','normal'),text],ensure_ascii=False)
    if key not in vector_cache:raise ValueError('Missing original font outline: '+key)
    w=vector_cache[key]['width'];path_data=vector_cache[key]['d']
    anchor=css.get('text-anchor','start')
    if anchor=='middle': x-=w/2
    elif anchor=='end': x-=w
    g=E.Element(tag('g'))
    if attrs.get('transform'): g.set('transform',attrs['transform'])
    g.append(E.Comment(' '+text+' '))
    if text.strip():
        E.SubElement(g,tag('path'),{'d':path_data,'transform':f'translate({x:.9f} {y:.9f}) scale(1 -1)',
            'fill':css.get('fill','#000000')})
    return g

def outline_all(root):
    for parent in list(root.iter()):
        for n in list(parent):
            if n.tag != tag('text'): continue
            if len(n):
                g=E.Element(tag('g'))
                if n.get('transform'):g.set('transform',n.get('transform'))
                for child in n:
                    if child.tag==tag('tspan'):
                        g.append(outline_piece(child.text or '',child.attrib,style(n.get('style',''))))
            else:g=outline_piece(n.text or '',n.attrib)
            i=list(parent).index(n);parent.remove(n);parent.insert(i,g)

def compose(root, number, positions, scale, width, height, letters):
    fig=by_id(root,'figure_1')
    axes=[by_id(root,f'axes_{i}') for i in range(1,5)]
    for n in list(fig):fig.remove(n)
    E.SubElement(fig,tag('rect'),{'width':str(width),'height':str(height),'fill':'white'})
    for i,(ax,(tx,ty)) in enumerate(zip(axes,positions)):
        # Remove only the titles/letters absent from the supplied assembled image.
        for n in list(ax):
            txt=''.join(n.itertext())
            if n.get('id','').startswith('text_') and any(s in txt for s in ['单蛋白体系','共表达体系',f'({chr(97+i)})']):
                ax.remove(n)
        bg=next(n for n in ax.iter(tag('path')))
        coords=list(map(float,re.findall(r'-?\d+(?:\.\d+)?',bg.get('d'))))
        left=min(coords[::2]);top=min(coords[1::2])
        wrap=E.SubElement(fig,tag('g'),{'id':f'panel_{chr(97+i)}','transform':f'translate({tx-scale*left:.9f} {ty-scale*top:.9f}) scale({scale})'})
        wrap.append(ax)
    for i,(x,y) in enumerate(letters):
        # Final composite panel letters are 16 px in the supplied references.
        fig.append(outline_piece(f'({chr(97+i)})',{'x':str(x),'y':str(y),'style':"font-size: 16px; font-family: 'DejaVu Sans'"}))
    root.set('width',f'{width}px');root.set('height',f'{height}px');root.set('viewBox',f'0 0 {width} {height}')

t4=parse(SRC/'图4_PINN先验模拟结果_完全沿用原四宫格指令_20261030.svg')
r4=t4.getroot()
g=by_id(r4,'text_59')
changed=0
for n in g.iter():
    if n.tag is E.Comment and 'MRE = 4.29%' in (n.text or ''):
        n.text=n.text.replace('4.29%','0.02%')
        continue
    if n.tag == tag('g') and n.get('transform','').startswith('translate(65.481313 382.743216)'):
        uses=list(n)
        for idx,old,new in [(6,'17','13'),(8,'15','13'),(9,'1c','15')]:
            key='{'+XL+'}href'
            assert uses[idx].get(key)=='#PingFangHK-Regular-'+old
            uses[idx].set(key,'#PingFangHK-Regular-'+new);changed+=1
assert changed==3
# Evaluate the same saved prior with energy_scale=0.24, matching correction.
# The source rectangles were plotted from unrounded values, not their labels.
prior_positive_024=101.7148881746153
prior_active_024=102.7720228087121
axis_height=567.060376-347.181192
for patch_id,label_id,old_value,new_value,old_top,center,old_baseline in [
    ('patch_20','text_72',101.71805665444364,prior_positive_024,383.735103,533.844214,374.471331),
    ('patch_21','text_73',103.39203363810539,prior_active_024,380.718114,740.045339,371.454342),
]:
    delta=(old_value-new_value)*axis_height/122.0
    bar=next(by_id(r4,patch_id).iter(tag('path')))
    bar.set('d',bar.get('d').replace(f'{old_top:.6f}',f'{old_top+delta:.9f}'))
    label=by_id(r4,label_id)
    for child in list(label):label.remove(child)
    label.append(outline_piece(f'{new_value:.2f}%',{
        'x':str(center),'y':str(old_baseline+delta),
        'style':"font-size: 11.2px; font-family: 'Hiragino Sans GB'; text-anchor: middle; fill: #334155"}))
compose(r4,4,[(82,24),(435,18),(82,360),(435,356)],1.164275,865,665,[(15,20),(373,20),(15,346),(373,346)])
t4.write(OUT/'图4_PINN先验模拟结果_统一0p24.svg',encoding='utf-8',xml_declaration=True)

t5=parse(SRC/'图4_物理轨迹校正结果_补充六蛋白共表达.svg')
r5=t5.getroot()
for n in r5.iter(tag('text')):
    if n.text=='MRE = 12.38%':n.text='MRE = 0.92%'
    if n.text=='103.39%':
        n.text=f'{prior_active_024:.2f}%'
        delta=(103.39203363810539-prior_active_024)*(564.524805-346.797976)/120.0
        newy=float(n.get('y'))+delta;n.set('y',f'{newy:.9f}')
        n.set('transform',f'rotate(-0 {n.get("x")} {newy:.9f})')
    if n.text=='31.00%':
        n.text='30.04%'
        delta=(31.0-30.043441711994017)*(246.643634-28.916805)/122.0
        newy=float(n.get('y'))+delta;n.set('y',f'{newy:.9f}')
        n.set('transform',f'rotate(-0 {n.get("x")} {newy:.9f})')
p=next(by_id(r5,'patch_5').iter(tag('path')))
top=246.643634-(246.643634-191.319604)*30.043441711994017/31
p.set('d',p.get('d').replace('191.319604',f'{top:.9f}'))
p=next(by_id(r5,'patch_15').iter(tag('path')))
delta=(103.39203363810539-prior_active_024)*(564.524805-346.797976)/120.0
p.set('d',p.get('d').replace('376.931308',f'{376.931308+delta:.9f}'))
compose(r5,5,[(60,24),(592,25),(60,351),(592,358)],1.166,858,656,[(0,21),(518,21),(0,348),(518,348)])
outline_all(r5)
t5.write(OUT/'图5_PINN物理校正结果_统一0p24.svg',encoding='utf-8',xml_declaration=True)

for root in [r4,r5]:assert not list(root.iter(tag('text')))
(OUT/'更新数值.json').write_text(json.dumps({
    'figure4_c_MRE_percent':0.017595609876396415,
    'sixprotein_NRMSR_evaluation_energy_scale':0.24,
    'figure4_d_positive_count':9197,
    'figure4_d_positive_mean_NRMSR_percent':prior_positive_024,
    'figure4_d_active_count':47,
    'figure4_d_active_mean_NRMSR_percent':prior_active_024,
    'figure5_c_prior_mean_NRMSR_percent':prior_active_024,
    'figure5_d_MRE_percent':0.924250797586275,
    'figure5_a_filtered_NRMSR_percent':30.043441711994017,
    'display_precision':'two decimal places',
    'unchanged':'all scatter images, concentration statistics, corrected-trajectory NRMSR, counts, axis/tick paths, panel font sizes and internal spacing',
    'fonts':'Both figures use the confirmed PingFang HK font; mathematical glyphs retain DejaVu Sans. All text is outlined.',
    'font_source':'Bundled original-font vector outlines; no system-font fallback',
},ensure_ascii=False,indent=2),encoding='utf-8')
print(OUT)

# The confirmed current manuscript Figure5 supersedes the historical
# intermediate template produced above. Font paths and layout stay frozen.
import subprocess,sys
subprocess.run([sys.executable,str(OUT/'prepare_current_figure5_data.py')],check=True)
subprocess.run([sys.executable,str(OUT/'rebuild_current_figure5.py')],check=True)
metadata=json.loads((OUT/'更新数值.json').read_text())
current=json.loads((OUT/'本图使用的新结果.json').read_text())
metadata.update({'figure5_d_MRE_percent':current['concentration']['MRE_percent'],
 'figure5_c_corrected_mean_NRMSR_percent':current['gt_1_uM']['mean_NRMSR_percent'],
 'figure5_c_corrected_count':current['gt_1_uM']['count'],
 'figure5_data_source':'current Figure4 prior followed by 8921 correction steps'})
(OUT/'更新数值.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
