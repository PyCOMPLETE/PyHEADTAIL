try:
    DYNAMIC_VERSIONING = True
    import os, subprocess
    worktree = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    gitdir = worktree + '/.git/'
    with open(os.devnull, 'w') as devnull:
        __version__ = subprocess.check_output(
            'git --git-dir=' + gitdir + ' --work-tree=' +
            worktree + ' describe --long --dirty --abbrev=10 --tags',
            shell=True, stderr=devnull)
    __version__ = __version__.decode('utf-8').rstrip() # remove trailing \n
    __version__ = __version__[1:] # remove leading v
    # remove commit hash to conform to PEP440:
    split_ = __version__.split('-')
    __version__ = split_[0]
    if split_[1] != '0':
        __version__ += '.' + split_[1]
    dirty = 'dirty' in split_[-1]
except:
    DYNAMIC_VERSIONING = False
    from ._version import __version__
    dirty = False

print ('PyHEADTAIL v' + __version__)
if dirty:
    print ('(dirty git work tree)')
print ('\n')

from .general.element import Element, Printing
from .general import utils
# print '                                                                                                                             '
# print '                                                                ;Cfttttt11111111tttt1f0f.                                    '
# print '                                                                ,GttttfG0GGG00000GGG0G0G0G0G0G0Ct1t1fG:                      '
# print '                                                           ,GtttLG000GGGGGGGGGGG00GGGGGGGGGGGGGG0GGG0GCtttLt                 '
# print '                                                       .CtttC0G00GGGGGGGGGGG00t1t1GGGGGGGGGGGGGGGGGG0GG0G00Gtttt             '
# print '                                                     G1tf000GGGGGGGGGG000811tGG0GL0GGGGGGGGGGGGGGGGGGGGGGGGGGGGG11C          '
# print '                                                  GtttGG0G00GGGGGGGGG08t1GGG0;;0G000GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGtt:       '
# print '                                                01tGG00GGGGGGGGGGGG0tt000ft::::GGG00GGGGGGGGGGGGGGGGGGG0GGG0G0000000GftG     '
# print '                              L1tLf.          8ttGGGGGGGGGGGGGG00GtGG8fft:::::;0GG@0GGGGGGGGGGGGG00G088888888888GttfLLLt1C   '
# print '                              CtG0G0GG8888t.G1tGGGGGGGGGGGGGGG0fGG0ffff;:::::::0G08GGGGGGGGGGGG888888888888f                 '
# print '                              .tt0GGGGG0888Ctf0GGGGGGGGGGGG0G0000GGGG0Gt;:;:::;0GG88GGGGG8808888888888C                      '
# print '                               C1GGGGGG0G01t00GGGGGGGGGGGGG00000000GGGGGGGG;::;GGG80GGGGGGGG0Lt1t@@i                         '
# print '                                1tGGGGG0811G00GGGGGGGGGGGGGGGG0880G0GGGGGG0G0;L0G08GGGGGGGGGGG0GGG1ttf                       '
# print '                                t1t0GG08t1G0GGGGGGGGGGGGG0GtLGGGG0GLtGGGGGGGGG8G08GGGGGGGGGGGGGGGGG00tttf                    '
# print '                                 0tCGG8tt00GGGGGGGGG0G0CtG0GG0GGG000GGt00GGGG000GGGGGGGGGGGGGGGGGGGGGG0G1tG                  '
# print '                                  0G0G81G0GGGGGGGGGG0Gf00GG8888@8@80GGG0CGGG0GGGGGGGGGGGGGGGGGGGGGGGGGG000ttL                '
# print '                                  C11GG080GGGGGGGG00t0GG08@111: .i1t80GG08GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGt1C              '
# print '                                  f0@tGGGGGGGGGGG00C0GG88i1         1GG0G0GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG0GGttC            '
# print '                                  G88ffG0GGGGGG0G0GGG0811.           .0GG08GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG0Gt1L           '
# print '                        C      .C .0G1800GGGGGG0G00f0811              .0G00GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG0G11i         '
# print '                      L         1  Lt 11GGGGGGG000ft81:     .8;        0G0GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG0GG001t0        '
# print '         L:     .C   1        f     t  8CG00GGGG0G181,      0G00       00G0GG0GGGGGGGGGGGGGGGGGGGGGGGG0GGGGGGGGGGGGtt8       '
# print '      f.           . .;iC   ..      C  .tGLG0GGGGt01.       0, Gi      0GGGGGGGGGGGGGGGGGGG0GGGGGGGGGGGGGGGGGGGGGGGGtt8      '
# print '      CL0t .G1111   :      G        1  08t8L0000t11:        00@8L     t0GGGGGGGGGGGGGGGGGGG80GGGGGGGGGGGGGGGGGGGGGGGG11L     '
# print '   1           C1   i     f          , :,0C10G0Ltf:         :C8@1    :00GGGGGGGGGGGGGGGGGGGGG8888880000GGGGGGGGGGGGGGCtt.    '
# print '   ,1G:C111.    .1   i    i          i  0@80ttt10i           G111   fGGG0GGGGGGGGGGGGGGGGGGGGG8888888888GGG0GGGGGGGGGGtt0    '
# print '  ;        tL1   01 ;              Gf C  00. LG1               f  f;;:::;;:8GGGGGGGGGGGGGGG0GGGt888888888880GGGGGGGG00Gtt.   '
# print '    i1i      C1   i111     .      C8@G  i0,G;::;C1             .t;:;:;::::::0GGGG00GGGGGGGGGGG0G11 .G8@8888880GGGGGGGGGL1G   '
# print ' L111tt1111.  1;  1111.   .         @888@@888:;:;;:;;;0.    :t::::;:   :::::8GG88GGGGG0GGGGGGGGGC1,      8888@8800GGGGGG1t   '
# print '  8       C1  f    L11i  ;. 1.          ;.:;:::::::::::::::::::::;;   ;;:::10G880GGGGGGGGGGGGGG0GL1         ,88888G0GGG0f1.  '
# print ' G  .i,    .1 G     ,1110f                L.::::::::::::::::::::;. i.L;;:;LG88888G0GGGGGGGGGGGGGG0tf           ,88880GGGGt;  '
# print '  t111111;  :11      f11L    G              .1.::::::::::::;CC;:;:;:::;;C88888888880GGGGGGGGGGGGG0G1G             0@8000G1t  '
# print '   L1Cf011. ;11     .11G    t1.L                ,L,.::::::;:::::;:::C@888888888888888G000GGGGGGGGGGt1               f880GtL  '
# print '    .G111111111.   .1t.    111..                .:LG8888@88088@0tfL@G       t88888888880GGGGGGGGG0GCti                CC0tL  '
# print '        ;L111111110.      .1C  i              :..::;ft888888888ffffffftt8.       08888880GGGGGGGGGGGtG                  tti  '
# print '             G11i,,,,.        0            :L..;:::::;ffC88888888fffi::::::..C;     C88888GGG0GGGG0GtC                   f.  '
# print '                .Gf11111.   :Cff.C     ,tftL.,;:::::::::;f888888888881:;:::::::..i.    888800GGGGGG01L                       '
# print '                 i1111111111ffft;:.....::ft..:::::::::::::0G000GGGG8888880f;;;::::..     88880GGGGGGtf                       '
# print '                      ..  iffff;:::::::;ttC.::::::::::::::0GGGG0GC888888800t G::::;.G      @88GGGGGGtf                       '
# print '                            . G::::::;1:  L.;:;:::::::::::GG0G0Gf888888  Gi10::::;:.       .;@8GGGGGtC                       '
# print '                                          ,.:::::::::::::;0GG1  L888 1L  fG:;:::::.           880GGG1C                       '
# print '                                           .,:::::::::::;80i......   1G   tf;:::.;             880001.                       '
# print '                                           :.:::::::::::t.        .        GftftC,G             G0Gtt                        '
# print '                                            :.:;::::::;i ,                   Ct11G               LC1i                        '
# print '                                              1:::::::1001         C              .               tf                         '
# print '                                                                                                                             '
