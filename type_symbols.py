import time, keyboard
from pynput.mouse import Button, Controller

mouse = Controller()

symbols = "ACTI.ST AMAST.ST BIM.ST CCOR-B.ST ERIC-A.ST G5EN.ST GIGSEK.ST OV.ST PREV-B.ST STAR-A.ST CE.ST ENEA.ST GOMX.ST MAHA-A.ST POOL-B.ST PREC.ST SVED-B.ST AOI.ST CLA-B.ST ENQ.ST NETI-B.ST SNM.ST STRAX.ST TIETOS.ST AGRO.ST ANOD-B.ST AUR.ST AZELIO.ST BEGR.ST BEIA-B.ST IMPC.ST KARO.ST KDEV.ST RATO-B.ST SALT-B.ST SSAB-B.ST BINV.ST BRG-B.ST ERIC-B.ST MIDW-B.ST MVIR-B.ST SAS.ST SSAB-A.ST VSSAB-B.ST"

sym_list = symbols.split()

for i in range(len(sym_list)):
	sym_list[i] = sym_list[i][:-3]

sym_str = "\n".join(sym_list)

with open('test_folder/syms.txt', 'w') as f:
	f.write(sym_str)



keyboard.wait('esc')
print('Starting in:')
for x in range(2):
	print('{}...'.format(x + 1))
	time.sleep(1)


count = 0
for sym in sym_list:
	count+=1
	print('{} iterated out of {}.'.format(count, len(sym_list)))
	keyboard.write(sym)
	time.sleep(1)
	keyboard.press_and_release('enter')
	time.sleep(0.3)
	mouse.press(Button.left)
	time.sleep(0.05)
	mouse.release(Button.left)
