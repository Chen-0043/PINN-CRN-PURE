#!/usr/bin/env python3
"""Update only Figure5(c,d) artwork; keep confirmed original outlined typography."""
from pathlib import Path
import copy,hashlib,json,re,xml.etree.ElementTree as E
import os,tempfile
os.environ.setdefault("MPLCONFIGDIR",tempfile.mkdtemp(prefix="pure_plot_cache_"))
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import Normalize,to_hex

ROOT=Path(__file__).resolve().parent
NS='http://www.w3.org/2000/svg';XL='http://www.w3.org/1999/xlink'
E.register_namespace('',NS);E.register_namespace('xlink',XL)
tag=lambda name:'{'+NS+'}'+name

def parse(path):return E.parse(path,parser=E.XMLParser(target=E.TreeBuilder(insert_comments=True)))
def by_id(root,value):return next(n for n in root.iter() if n.get('id')==value)
def asbytes(node):return E.tostring(node,encoding='utf-8')
def comment_group(node,text):
    return next(g for g in node.iter() if any(c.tag is E.Comment and (c.text or '').strip()==text for c in list(g)))

def main():
    tree=parse(ROOT/'原图5_版式与字形模板.svg');root=tree.getroot();before=copy.deepcopy(root)
    cache=json.loads((ROOT/'新标签原字体轮廓.json').read_text())
    metrics=json.loads((ROOT/'本图使用的新结果.json').read_text())
    value=metrics['gt_1_uM']['mean_NRMSR_percent'];count=metrics['gt_1_uM']['count']
    assert f'{value:.2f}%'=='23.09%' and count==34
    # Preserve the original bar position/width and the exact label-to-bar gap.
    bar=next(by_id(root,'patch_16').iter(tag('path')))
    old_top=520.588427;bottom=564.524805;top=346.797976
    new_top=bottom-value*(bottom-top)/120
    assert bar.get('d').count(f'{old_top:.6f}')==2
    bar.set('d',bar.get('d').replace(f'{old_top:.6f}',f'{new_top:.9f}'))
    g=comment_group(by_id(root,'text_50'),'24.22%')
    path=next(g.iter(tag('path')))
    shift=new_top-old_top;baseline=511.623988+shift
    m=re.fullmatch(r'translate\(([-\d.]+) ([-\d.]+)\) scale\(1 -1\)',path.get('transform'))
    assert m
    key=json.dumps([11.2,'Hiragino Sans GB','normal','23.09%'],ensure_ascii=False)
    path.set('d',cache[key]['d']);path.set('transform',f'translate({float(m[1]):.9f} {baseline:.9f}) scale(1 -1)')
    g.set('transform',f'rotate(-0 346.508866 {baseline:.9f})')
    next(c for c in g if c.tag is E.Comment).text=' 23.09% '
    # Change only the glyph path for the count; its position, fontsize, line
    # spacing and exact string advance remain unchanged.
    g=comment_group(by_id(root,'text_39'),'33条校正轨迹')
    key=json.dumps([9.8,'Hiragino Sans GB','normal','34条校正轨迹'],ensure_ascii=False)
    next(g.iter(tag('path'))).set('d',cache[key]['d'])
    next(c for c in g if c.tag is E.Comment).text=' 34条校正轨迹 '
    # Rebuild the actual newly calculated scatter, preserving source limits,
    # marker size9, viridis mapping, alpha0.60 and layer order.
    data=pd.read_csv(ROOT/'六蛋白6h新接续浓度_9198对象.csv')
    target=data.teacher_6h_uM.to_numpy(float);prediction=data.new_corrected_6h_uM.to_numpy(float)
    assert len(target)==len(prediction)==9198
    rel=abs(prediction-target)/np.maximum(abs(target),1e-9)
    color=np.clip(np.log10(rel+1e-6),-6,1)
    colors=colormaps['viridis'](Normalize(vmin=color.min(),vmax=color.max())(color))
    left=607.650533;right=825.377362;upper=346.797976;lower=564.524805
    x=left+(np.log10(np.maximum(target,1e-12))+12)/17*(right-left)
    y=lower-(np.log10(np.maximum(prediction,1e-12))+12)/17*(lower-upper)
    axis=by_id(root,'axes_4');oldimage=next(c for c in axis if c.tag==tag('image'));position=list(axis).index(oldimage);axis.remove(oldimage)
    scatter=E.Element(tag('g'),{'id':'new_sixprotein_corrected_scatter','clip-path':'url(#pdd9f17b337)'})
    # Matplotlib's circular marker scaled to s=9 points^2 (radius1.5).
    marker=E.SubElement(scatter,tag('defs'))
    circle='M 0 1.5 C 0.397805 1.5 0.77937 1.341951 1.06066 1.06066 C 1.341951 0.77937 1.5 0.397805 1.5 0 C 1.5 -0.397805 1.341951 -0.77937 1.06066 -1.06066 C 0.77937 -1.341951 0.397805 -1.5 0 -1.5 C -0.397805 -1.5 -0.77937 -1.341951 -1.06066 -1.06066 C -1.341951 -0.77937 -1.5 -0.397805 -1.5 0 C -1.5 0.397805 -1.341951 0.77937 -1.06066 1.06066 C -0.77937 1.341951 -0.397805 1.5 0 1.5 z'
    E.SubElement(marker,tag('path'),{'id':'sixprotein_marker_s9','d':circle})
    for px,py,c in zip(x,y,colors):
        E.SubElement(scatter,tag('use'),{'{'+XL+'}href':'#sixprotein_marker_s9','x':f'{px:.9f}','y':f'{py:.9f}',
                                      'style':f'fill: {to_hex(c)}; fill-opacity: 0.6; stroke: none'})
    axis.insert(position,scatter)
    # Display precision stays identical to the confirmed original: two places
    # for MRE and four forR². The new raw values round to the same strings.
    assert f"{metrics['concentration']['MRE_percent']:.2f}"=='0.92'
    assert f"{metrics['concentration']['R2']:.4f}"=='0.9863'
    assert not list(root.iter(tag('text')))
    # Structural checks detect even subtle font, tick, title, size or spacing
    # changes outside the two intentionally updated c-panel labels/scatter.
    assert root.attrib==before.attrib
    for name in ['panel_a','panel_b']:
        assert asbytes(by_id(root,name))==asbytes(by_id(before,name)),name
    for name in ['panel_a','panel_b','panel_c','panel_d']:
        assert by_id(root,name).attrib==by_id(before,name).attrib,name
    old_c=copy.deepcopy(by_id(before,'axes_3'));new_c=copy.deepcopy(by_id(root,'axes_3'))
    old_d=copy.deepcopy(by_id(before,'axes_4'));new_d=copy.deepcopy(by_id(root,'axes_4'))
    for axis_copy in [old_c,new_c]:
        for id in ['patch_16','text_50','text_39']:
            node=by_id(axis_copy,id)
            for parent in axis_copy.iter():
                if node in list(parent):parent.remove(node);break
    assert asbytes(old_c)==asbytes(new_c),'Unintended panelc typography/layout change'
    for axis_copy in [old_d,new_d]:
        for child in list(axis_copy):
            if child.tag==tag('image') or child.get('id')=='new_sixprotein_corrected_scatter':axis_copy.remove(child)
    assert asbytes(old_d)==asbytes(new_d),'Unintended paneld typography/layout change'
    output=ROOT/'图5_PINN物理校正结果_统一0p24.svg'
    tree.write(output,encoding='utf-8',xml_declaration=True)
    audit={'dimensions':root.attrib,'deGFP_panels_a_b_identical':True,'all_panel_transforms_identical':True,
           'panel_c_axis_fonts_ticks_and_spacing_identical':True,'panel_d_all_typography_and_axes_identical':True,
           'all_text_outlined_no_font_substitution':True,'new_scatter_points':len(target),
           'new_bar_mean_NRMSR_percent':value,'new_bar_trajectory_count':count,
           'raw_MRE_percent':metrics['concentration']['MRE_percent'],'display_MRE_percent':'0.92%',
           'raw_R2':metrics['concentration']['R2'],'display_R2':'0.9863',
           'source_sha256':hashlib.sha256((ROOT/'原图5_版式与字形模板.svg').read_bytes()).hexdigest(),
           'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}
    (ROOT/'图5_数据与格式核查.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print(output)

if __name__=='__main__':main()
