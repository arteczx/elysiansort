import astroquery
import exoplanet
import lightkurve as lk
import matplotlib.pyplot as plt

def get_exoplanet_info(planet_name):
    try:
        planet = exoplanet.Exoplanet(planet_name)
    except ValueError:
        print(f"Eksoplanet dengan nama '{planet_name}' tidak ditemukan.")
        return

    planet_data = planet_data[0]
    print("Informasi Eksoplanet:")
    print(f"Nama: {planet_data['NAME']}")
    print(f"Metode Penemuan: {planet_data['DISCOVERYMETHOD']}")
    print(f"Tahun Penemuan: {planet_data['DISCOVERYYEAR']}")
    print(f"Periode Orbit (hari): {planet_data['PER']}")
    print(f"Jari-jari (jari bumi): {planet_data['RADIUS']}")
    print(f"Massa (massa Jupiter): {planet_data['MASS']}")
    print(f"Jarak ke Bintang (parallax, mas): {planet_data['PARALLAX']}")
    print(f"Eksentrisitas Orbit: {planet_data['ECCENTRICITY']}")
    print(f"Suhu Permukaan (K): {planet_data['TEQ']}")
    print(f"Jumlah Observasi: {planet_data['NCONFIRM']}")  # You can use other relevant columns
    print(f"Sumber Data: {planet_data['SOURCEDISCOVERY']}")
    print("-" * 50)

    # Get information about the host star
    star_name = planet_data['STAR_NAME']
    get_host_star_info(star_name)

    # Analyze the exoplanet light curve using lightkurve
    analyze_exoplanet_light_curve(planet_name)


def get_host_star_info(star_name):
    try:
        star_data = exoplanet.get_star_info(star_name)
    except ValueError:
        print(f"Bintang induk dengan nama '{star_name}' tidak ditemukan.")
        return

    print("Informasi Bintang Induk:")
    print(f"Nama: {star_data['name']}")
    print(f"Tipe Spektral: {star_data['type']}")
    print(f"Jarak ke Bumi (parsecs): {star_data['distance']}")
    print(f"Massa (massa Matahari): {star_data['mass']}")
    print(f"Jari-jari (radius Matahari): {star_data['radius']}")
    print(f"Suhu Efektif (K): {star_data['teff']}")
    print(f"Kecepatan Rotasi (km/s): {star_data['rotation']}")
    print(f"Metalisitas ([Fe/H]): {star_data['metallicity']}")
    print(f"Usia (miliar tahun): {star_data['age']}")
    print("-" * 50)



def analyze_exoplanet_light_curve(planet_name):
    search_result = lk.search_lightcurve(planet_name, mission="TESS")
    
    if len(search_result) == 0:
        print(f"Tidak ditemukan data kurva cahaya untuk eksoplanet dengan nama '{planet_name}'.")
        return

    lc = search_result.download()

    # Plot and analyze the light curve
    lc.plot()
    periodogram = lc.to_periodogram(method="bls")
    best_fit_period = periodogram.period_at_max_power
    print("Periode terbaik (hari):", best_fit_period)

    folded_lc = lc.fold(period=best_fit_period.value)
    folded_lc.scatter()
    print("Jari-jari (jari bumi) dari kurva cahaya transit:", folded_lc.depth)

    plt.show()


def main():
    planet_name = input("Masukkan nama eksoplanet: ")
    get_exoplanet_info(planet_name)


if __name__ == "__main__":
    main()
