# main.py
from SleepPredictor import SleepPredictor
from SleepPlotter import SleepPlotter

# SleepPredictor sınıfını kullanarak tahmin işlemleri
predictor = SleepPredictor()
predictor.process_data()
predictor.train_model()

# Sonuçları alın
results = predictor.get_results()

# SleepPlotter sınıfı ile grafik çizimleri
plotter = SleepPlotter(results)
plotter.plot_grouped_barplot()  # Tahmini ve gerçek uyku süreleri karşılaştırma grafiği
