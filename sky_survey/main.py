import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
from astropy.coordinates import SkyCoord, get_sun
from astropy.time import Time
from astropy import units as u
from astropy.io import fits
import astropy.visualization as vis
import os

# Mengatur gaya plot menjadi gaya AstroPy
plt.style.use(astropy_mpl_style)

# Mendapatkan input dari pengguna
coordinate_input = input("Masukkan koordinat langit (ra dec dalam format J2000): ")
ra, dec = map(float, coordinate_input.split())

# Membuat objek SkyCoord berdasarkan koordinat pengguna
target_coord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')

# Mendapatkan koordinat Matahari saat ini
current_time = Time.now()
sun_coord = get_sun(current_time)

# Mendapatkan jarak sudut antara target dan Matahari
angular_separation = target_coord.separation(sun_coord)

# Mendapatkan jumlah panjang gelombang yang diinginkan dari pengguna
num_wavelengths = int(input("Masukkan jumlah panjang gelombang: "))

# Meminta panjang gelombang dari pengguna
wavelengths = []
for i in range(num_wavelengths):
    wavelength_input = float(input(f"Masukkan panjang gelombang ke-{i+1} (dalam um): "))
    wavelengths.append(wavelength_input * u.um)

# Membuat direktori untuk menyimpan file FITS
fits_dir = "fits_files"
os.makedirs(fits_dir, exist_ok=True)

# Membuat gambar plot
fig, axs = plt.subplots(1, num_wavelengths, figsize=(15, 5))

for i, wavelength in enumerate(wavelengths):
    # Mengecek apakah file FITS sudah ada
    fits_filename = f"wavelength_{wavelength.value}_ra_{ra}_dec_{dec}.fits"
    fits_path = os.path.join(fits_dir, fits_filename)
    if os.path.exists(fits_path):
        print(f"Using existing FITS file: {fits_path}")
        hdu = fits.open(fits_path)
        image_data = hdu[0].data
    else:
        # Mengambil gambar FITS dari pustaka AstroPy
        url = f"http://archive.stsci.edu/cgi-bin/dss_search?v=poss2ukstu_red&r={target_coord.ra.deg}&d={target_coord.dec.deg}&e={wavelength.value}&h={wavelength.value}&w={wavelength.value}&f=fits"
        hdu = fits.open(url)

        # Mengambil data gambar dari HDU
        image_data = hdu[0].data

        # Menyimpan file FITS
        hdu.writeto(fits_path, overwrite=True)
        print(f"Saved FITS file: {fits_path}")

    # Menentukan skala dan pemrosesan gambar
    scaled_image = vis.sqrt_stretch(vis.AsinhStretch(0.2))(image_data)

    # Memplot gambar
    axs[i].imshow(scaled_image, cmap='gray')
    axs[i].set_title(f'Wavelength: {wavelength}')

    # Menutup file FITS
    hdu.close()

plt.show()
