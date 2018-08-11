from PIL import Image, ImageDraw, ImageFont
import os
import random
from math import sin, cos, radians

chars = 'ABCČĆDĐEFGHIJKLMNOPRSŠTUVZŽ0123456789'
special='ČĆĐ'

def generate_chars(folder, n):
	os.mkdir('./' + folder)
	print('Generating ' + folder + ' (' + str(n) + ' examples per character)...')
	for c in chars:
		fonts = os.listdir('./fonts/')
		
		if c in special:
			fonts.remove('Helvetica-Regular.ttf')
		if c=='0':
			fonts.remove('Targa MS.ttf')
			fonts.remove('Dosis-Medium.ttf')
		os.mkdir('./' + folder + '/' + c)
		
		for i in range(1, (n + 1)):
			color = random.randint(100, 255)
			fill = random.randint(0, 60)
			img = Image.new('RGB', (50, 50), (color, color, color))
			font = ImageFont.truetype('./fonts/' + random.choice(fonts), 24)
			draw = ImageDraw.Draw(img)
			# Velicina znaka koji cemo nacrtati
			(w, h) = font.getsize(c)
			# Nacrtajmo znak na sredini slike
			draw.text((25-w//2, 25-h//2), c, (fill, fill, fill), font)
			# Koordinate okvira oko znaka (u koordinatnom sustavu s ishodistem u sredini slike):
			x1, y1 = -w//2, -h//2
			x2, y2 = +w//2, -h//2
			x3, y3 = +w//2, +h//2
			x4, y4 = -w//2, +h//2
			# Rotiranje slike oko centra
			fi_degrees = random.randint(-20, 20)
			img = img.rotate(fi_degrees)
			# Koordinate okvira oko rotiranog znaka
			# (vracamo ishodiste u gornji lijevi kut slike, zato svuda +25)
			fi = radians(fi_degrees)
			xr1, yr1 = x1*cos(fi) - y1*sin(fi) + 25, x1*sin(fi) + y1*cos(fi) + 25
			xr2, yr2 = x2*cos(fi) - y2*sin(fi) + 25, x2*sin(fi) + y2*cos(fi) + 25
			xr3, yr3 = x3*cos(fi) - y3*sin(fi) + 25, x3*sin(fi) + y3*cos(fi) + 25
			xr4, yr4 = x4*cos(fi) - y4*sin(fi) + 25, x4*sin(fi) + y4*cos(fi) + 25
			# Koordinate gornjeg lijevog i donjeg desnog kuta rotiranog znaka
			xr1, yr1 = min(xr1, xr4), min(yr1, yr2)
			xr3, yr3 = max(xr2, xr3), max(yr3, yr4)
			# Izrezivanje slike samog znaka
			bijeli_rub_hor = random.randint(0,3)
			bijeli_rub_vert = random.randint(0,3)
			pomak_hor = random.randint(0,bijeli_rub_hor)
			pomak_vert = random.randint(0,bijeli_rub_vert)
			# Gornji lijevi vrh izrezane slike
			x_ul = xr1 - pomak_hor
			y_ul = yr1 - pomak_vert
			# Donji desni vrh izrezane slike
			x_dr = xr3 + bijeli_rub_hor - pomak_hor
			y_dr = yr3 + bijeli_rub_vert - pomak_vert
			img = img.crop((x_ul, y_ul, x_dr, y_dr))
			img = img.resize((24,32))
			img.save('./' + folder + '/' + c + '/' + c + '_' + str(i) + '.jpg')
	print('Finished.')

generate_chars('training data', 3000)
generate_chars('validation data', 300)

