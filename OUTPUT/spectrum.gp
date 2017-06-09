set pointsize 0.01
set xlabel "{/Symbol q} (in units of {/Symbol p})"
set xrange [0:2]
set ylabel "{/Symbol x}({/Symbol q})"
set yrange [0:15]
plot "ee120" using ($1/60.0):2:(0) notitle w err lt 1 lw 2
replot "ee120" using ($1/60.0):3:(0) notitle w err lt 1 lw 2
replot "ee120" using ($1/60.0):4:(0) notitle w err lt 1 lw 2
replot "ee120" using ($1/60.0):5:(0) notitle w err lt 1 lw 2
replot "ee120" using ($1/60.0):6:(0) notitle w err lt 1 lw 2
replot "ee120" using ($1/60.0):7:(0) notitle w err lt 1 lw 2
replot "ee120" using ($1/60.0):8:(0) notitle w err lt 1 lw 2
replot "ee120" using ($1/60.0):9:(0) notitle w err lt 1 lw 2
replot "ee120" using ($1/60.0):10:(0) notitle w err lt 1 lw 2
replot "ee120" using ($1/60.0):11:(0) notitle w err lt 1 lw 2
replot "ee120" using ($1/60.0):12:(0) notitle w err lt 1 lw 2
replot "ee120" using ($1/60.0):13:(0) notitle w err lt 1 lw 2
replot "ee120" using ($1/60.0):14:(0) notitle w err lt 1 lw 2
replot "ee120" using ($1/60.0):15:(0) notitle w err lt 1 lw 2
replot "ee120" using ($1/60.0):16:(0) notitle w err lt 1 lw 2
replot "ee120" using ($1/60.0):17:(0) notitle w err lt 1 lw 2
replot "ee120" using ($1/60.0):18:(0) notitle w err lt 1 lw 2
replot "ee120" using ($1/60.0):19:(0) notitle w err lt 1 lw 2
replot "ee120" using ($1/60.0):20:(0) notitle w err lt 1 lw 2
replot "ee120" using ($1/60.0):21:(0) notitle w err lt 1 lw 2
replot "ee120" using ($1/60.0):22:(0) notitle w err lt 1 lw 2
replot "ee120" using ($1/60.0):23:(0) notitle w err lt 1 lw 2
replot "ee120" using ($1/60.0):24:(0) notitle w err lt 1 lw 2
replot "ee120" using ($1/60.0):25:(0) notitle w err lt 1 lw 2
replot "ee120" using ($1/60.0):26:(0) notitle w err lt 1 lw 2
replot "ee120" using ($1/60.0):27:(0) notitle w err lt 1 lw 2
replot "ee120" using ($1/60.0):28:(0) notitle w err lt 1 lw 2
set term post eps color enhanced 24
set output "entanglementSpectrum.eps"
replot
set term x11

reset
set pointsize 0.01
set xlabel "{/Symbol q} (in units of {/Symbol p})"
set xrange [0:2]
set ylabel "E({/Symbol q})"
plot "data120" using ($1/60.0):2:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):3:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):4:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):5:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):6:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):7:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):8:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):9:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):10:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):11:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):12:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):13:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):14:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):15:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):16:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):17:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):18:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):19:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):20:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):21:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):22:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):23:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):24:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):25:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):26:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):27:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):28:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):29:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):30:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):31:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):32:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):33:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):34:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):35:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):36:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):37:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):38:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):39:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):40:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):41:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):42:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):43:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):44:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):45:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):46:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):47:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):48:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):49:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):50:(0) notitle w lines lt 1 lw 2
replot "data120" using ($1/60.0):51:(0) notitle w lines lt 1 lw 2
set term post eps color enhanced 24
set output "energySpectrum.eps"
replot
set term x11

