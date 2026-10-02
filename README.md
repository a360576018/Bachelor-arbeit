# Bachelorarbeit

Dieses Repository enthält den Code und die Versuchsergebnisse zu meiner Bachelorarbeit über Anomalieerkennung mit PatchCore.

## Dateien

- `main.py`: Startet die Hauptversuche für verschiedene Kategorien, Trainingsbildanzahlen und Seeds sowie die anschließende Auswertung.
- `data_sampler.py`: Wählt die fehlerfreien Trainingsbilder aus und speichert die Bildauswahl.
- `patchcore_runner.py`: Trainiert und bewertet ein PatchCore-Modell für eine Versuchskonfiguration.
- `result_process.py`: Speichert die Ergebnisse, berechnet Mittelwerte und Standardabweichungen und erstellt Diagramme.
- `knee_analysis.py`: Bestimmt die Kategorieeinteilung, den Kneedle-Punkt und die empfohlene Trainingsbildanzahl.
- `sensitivity_analysis.py`: Untersucht, wie verschiedene Werte von rho und k die Empfehlungen beeinflussen.
- `train_export.py`: Trainiert und speichert die Modelle mit der empfohlenen und der vollständigen Trainingsbildanzahl für den Hardwarevergleich.

## Ordner docker_evaluation

- `evaluate.py`: Lädt die gespeicherten Modelle und misst Erkennungsleistung, Inferenzzeit sowie RAM- und VRAM-Verbrauch.
- `Dockerfile`: Erstellt das Docker-Image für die Hardware-Evaluation.
- `requirements.txt`: Enthält die zusätzlichen Python-Pakete für die Docker-Umgebung.
- `README.md`: Enthält die Befehle zum Erstellen und Starten des Docker-Containers.

## Ordner results

- `results_raw.csv`: Enthält die Ergebnisse der einzelnen Versuchsläufe.
- `results_mean.csv`: Enthält die zusammengefassten Ergebnisse über die Seeds.
- `recommendations.csv`: Enthält die empfohlenen Trainingsbildanzahlen und den Vergleich mit der vollständigen Trainingskonfiguration.
- `sensitivity_analysis.csv`: Fasst die Ergebnisse für jede Kombination von rho und k zusammen.
- `sensitivity_recommendations.csv`: Enthält die empfohlenen Trainingsbildanzahlen je Kategorie für die verschiedenen Parameterkombinationen.
- `hardware_evaluation.csv`: Enthält die Ergebnisse des Hardwarevergleichs.
- `plots/`: Enthält die Diagramme zur Erkennungsleistung und zum Speicherbedarf der Memory Bank.
- `knee_plots/`: Enthält die Diagramme zur Kneedle- und Stabilitätsanalyse.

## Hinweise

Der MVTec-AD-Datensatz muss separat heruntergeladen werden. Vor dem Start müssen die Datenpfade angepasst und die benötigten Python-Pakete installiert werden.

`main.py` startet die Hauptversuche. Für den Hardwarevergleich werden die Modelle zuerst mit `train_export.py` gespeichert und anschließend mit `docker_evaluation/evaluate.py` ausgewertet.
