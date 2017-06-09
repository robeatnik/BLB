set key bottom
set xlabel "{/Symbol q}'_c (in units of {/Symbol p})"
set xrange [0:2]
set ylabel "P({/Symbol q}'_c)"
plot "< grep Test log_scanE_cutoff16 | cut -c16-" w linespoints title "16"
replot "< grep Test log_scanE_cutoff64 | cut -c16-" w linespoints title "64"
replot "< grep Test log_scanE_cutoff256 | cut -c16-" w linespoints title "256"
replot "< grep Test log_scanE | cut -c16-" w linespoints title "All (729)"

#replot "< grep Test log_scanE_cutoff128 | cut -c16-" w linespoints
#replot "< grep Test log_scanE_cutoff32 | cut -c16-" w linespoints

set term post eps color enhanced 24
set output "scanE.eps"
replot
set term x11

plot "< grep Test log_scanES | cut -c16-" w linespoints notitle

set term post eps color enhanced 24
set output "scanES.eps"
replot
set term x11
