# Audyt AANCA: kod, logika badania, odtwarzalność i strona

Data: 2 października 2026. Badany commit: `fa3311bf750c32b7f2bc0ac69097018620e77d71`.

Ten raport zachowuje stan sprzed napraw. Ich implementację i ponowną weryfikację
opisuje [`raport napraw V1`](v1_remediation_2026-10-02.md).

## Ocena

Projekt ma działający, dobrze testowany mechanizm badawczy i rozbudowane zabezpieczenia pochodzenia danych. Nie znalazłem podstaw do odrzucenia wszystkich opublikowanych wyników ani do przepisywania V1 od zera. Znalazłem natomiast konkretne błędy opisu statystycznego, walidacji wejścia i publikacji, które warto poprawić przed kolejnym udostępnieniem projektu.

Najpilniejsze są: poprawne nazwanie przedziału H4, przywrócenie działania oficjalnej domeny i ujednolicenie wdrożenia z aktualną wersją repozytorium. Następnie należy naprawić walidację prawdopodobieństw, etykiet i progów oraz jawnie przenosić informację o zbieżności modeli. Dużą redukcję kodu należy realizować w oddzielnym V2, zgodnie z istniejącą decyzją D047.

To jest raport z przeglądu. Nie wdrożono opisanych poprawek produkcyjnych. Nie zmieniono źródłowych adnotacji, zamrożonych konfiguracji, wyników ani etapów ukończenia. Syntetyczne przykłady błędów są testami oprogramowania, a nie nowymi wynikami badania.

## Zakres i wykonane sprawdzenia

Przegląd objął dokumenty naukowe i operacyjne, strukturę całego repozytorium, inwentaryzację i analizę AST modułów Python, zależności między modułami, szczegółowe przejście kluczowych ścieżek danych, podziałów, OOF, korupcji, rankingów, odtwarzania etykiet, statystyki, walidacji zewnętrznej, CLI i generowania strony. Uruchomiono cały zestaw testów oraz weryfikatory zapisanych dowodów. Stronę porównano na poziomie tekstu, JSON, manifestów i zachowania przeglądarki.

Inwentaryzacja śledzonych plików Python:

| Obszar | Pliki | Fizyczne linie, łącznie z komentarzami i pustymi liniami |
|---|---:|---:|
| `src/` | 111 | 121 908 |
| `tests/` | 99 | 43 857 |
| `scripts/` | 20 | 6 223 |
| Razem | 230 | 171 988 |

Nie oznacza to ręcznego przeczytania każdej z tych linii ani formalnego dowodu poprawności wszystkich ścieżek. Nie przeprowadzono ponownie pełnych eksperymentów od obrazów do wyników. Audyt nie obejmuje oddzielnego repozytorium AANCA-V2.

Dowody z prób: [project_audit_2026-10-02_evidence.json](project_audit_2026-10-02_evidence.json). Zawierają rzeczywiste odpowiedzi HTTP, skróty plików, wyniki kontrprzykładów i odczyt tablic H4. Priorytet P1 oznacza poprawkę przed kolejną publikacją; P2 oznacza istotną poprawkę poprawności lub odtwarzalności; P3 oznacza doprecyzowanie dokumentacji.

## Ustalenia

### F01 · P1 · Przedział H4 jest opisany jako inna statystyka niż obliczana

**Lokalizacja:** `src/histo_audit/experiment/primary_core.py:1822`, `src/histo_audit/mvp_demo.py:1338`, `src/histo_audit/mvp_demo.py:2022`, `src/histo_audit/mvp_demo.py:2509`.

Kod zapisuje jeden wynik modelu po kierowanej restauracji, powiela go 100 razy i odejmuje wyniki 100 modeli po losowej restauracji. Następnie bierze kwantyle 2,5% i 97,5% tych różnic. Zbiór końcowy jest stały; nie następuje w tym obliczeniu bootstrap jego grup.

Potwierdza to zapisany artefakt: `pairing = same_final_reference_set_across_frozen_random_review_repetitions`, 100 elementów i dokładnie jedna unikalna wartość po stronie kierowanej. Odczyt dał różnicę średnią `-0.0021560596665870235` i kwantyle `[-0.0028586383107464695, -0.0013925186647962915]`, zgodne z publikowanymi liczbami.

Tymczasem artykuł ogólnie przypisuje wyświetlane przedziały sparowanemu bootstrapowi całych grup, a wykres H4 nazywa ten przedział `95% CI`. Ta rozbieżność występuje zarówno lokalnie, jak i na dostępnej stronie Hostinger.

**Skutek:** czytelnik może odczytać zmienność losowych strategii przeglądu na jednym zbiorze jako przedział ufności uwzględniający zmienność grup testowych. Zgodność liczb i checksum tego nie wychwytuje. Ujemny wynik punktowy pozostaje ujemny; ustalenie nie stanowi dowodu korzystnego efektu AANCA.

**Poprawka:** zachować zamrożone liczby i opisać przedział jako centralny 95-procentowy zakres różnic względem 100 losowych powtórzeń na stałym zbiorze. Przenosić typ przedziału i jednostkę resamplingu do `evidence.json`, podpisów i tekstów dostępności. Ewentualne nowe obliczenie bootstrapu grup publikować oddzielnie, z jawną kwalifikacją analizy po poznaniu wyników. Test powinien sprawdzać semantykę przedziału, nie tylko jego wartości.

### F02 · P1 · Oficjalna domena jest niedostępna w sprawdzonym środowisku, a dostępne wdrożenie jest starsze

**Adresy:** [aancastudy.org](https://aancastudy.org/), [www.aancastudy.org](https://www.aancastudy.org/), [techniczny adres Hostinger](https://mediumaquamarine-wombat-125861.hostingersite.com/).

2 października 2026 oficjalne adresy HTTPS oraz HTTP zwracały `403 Forbidden`, z serwera `hcdn`. Potwierdzono to żądaniem HTTP i w przeglądarce. Nie ustalono przyczyny po stronie konfiguracji serwera ani dostępności z innych sieci.

Adres techniczny zwraca `200`, ale artykuł ma datę **22 August 2026**, podczas gdy lokalny pakiet ma **26 August 2026**. Zdalne `evidence.json` ma `schema_version: 3`, lokalne `5`. Zdalnie brakuje sekcji `independent_pathologist_validation` i `public_independent_pathologist_replication`: odrębnego wyniku NuCLS JP.1 oraz replikacji RIVA i MIDOG++. Przeglądarka również nie znajduje na tej stronie nazw RIVA ani MIDOG++.

**Skutek:** opis dostępny odbiorcy nie odzwierciedla aktualnego stanu dowodów w repozytorium. Jest to materialna niespójność, choć starsza strona nie wyolbrzymia nowych pozytywnych wyników — po prostu ich nie zawiera.

**Poprawka:** ustalić konfigurację domeny i opublikować jeden zweryfikowany pakiet po naprawieniu F01. Sprawdzenie po wdrożeniu musi pobrać stronę, JSON i manifest z docelowej domeny; sam lokalny `present_demo.py --verify-only` nie weryfikuje wdrożenia. Audyt nie wykonywał operacji na hostingu.

### F03 · P2 · Pliki graficzne wdrożenia nie odpowiadają jego własnemu manifestowi

**Lokalizacja:** zdalny `manifest.json` i osiem adresów PNG na technicznym adresie Hostinger.

Spośród 12 plików opisanych przez zdalny manifest osiem serwowanych PNG miało inne SHA-256. Dotyczy to sześciu sprite'ów jąder, grafiki metody i overlayów. HTML, JSON, README i skrypt findings zgadzały się z manifestem zdalnym.

Przykład: `nucleus-compact.png` według zdalnego manifestu powinien mieć 1 570 355 bajtów i skrót zaczynający się od `fa5f3f61`; odpowiedź HTTP zawierała 1 538 687 bajtów i skrót `90c2da22...`. Był to PNG z `Content-Type: image/png`, bez `Content-Encoding`.

**Skutek:** nie można potwierdzić integralności całej publikacji na podstawie dostępnych URL-i i jej manifestu. Obrazy wyświetlały się poprawnie. Nie jest to dowód zmiany wyników naukowych ani celowej ingerencji; możliwa transformacja CDN wymaga osobnego sprawdzenia.

**Poprawka:** uzgodnić sposób serwowania z kontraktem integralności. Dla plików objętych SHA-256 zapewnić dostęp do nieprzekształconych bajtów albo opublikować osobny, dokładnie zdefiniowany pakiet archiwalny. Po wdrożeniu porównywać wszystkie pliki, a nie wyłącznie kod odpowiedzi strony głównej.

### F04 · P2 · Metryki przyjmują nieprawidłowe prawdopodobieństwa

**Lokalizacja:** `src/histo_audit/evaluation/restoration.py:282`.

`classification_metrics()` sprawdza skończoność i sumę wiersza, ale nie zakres `[0, 1]`. Dla etykiet `[0, 1]`, klas `(0, 1)` i macierzy `[[1.1, -0.1], [-0.1, 1.1]]` funkcja zwróciła accuracy `1.0` i ECE `0.0`. Pewności większe od 1 wypadają poza przedziały użyte do ECE, więc błędne wejście pozornie wygląda na idealnie skalibrowane.

**Skutek:** uszkodzone tablice lub błędny adapter modelu mogą przejść ocenę metryk, a także funkcje decyzyjne korzystające z tego walidatora. Nie wykazano takich wartości w opublikowanych tablicach; inne walidatory OOF mają silniejsze kontrole.

**Poprawka:** jeden ścisły walidator kształtu, skończoności, zakresu, sum wierszy i jednoznacznej kolejności klas w utrzymywanym rdzeniu. Nie normalizować ani nie przycinać błędnych danych bez jawnej decyzji. Dodać test wartości ujemnych, wartości ponad 1, NaN oraz powtórzonych klas.

### F05 · P2 · Ujemny próg omija zasadę przyjmowania tylko poprawy

**Lokalizacja:** `src/histo_audit/evaluation/retraining_guard.py:135`, `src/histo_audit/evaluation/retraining_guard.py:170`, `src/histo_audit/auditing/two_queue.py:319`, `src/histo_audit/auditing/two_queue.py:388`.

`evaluate_retraining_guard()` wymaga wyłącznie skończonego `minimum_effect`. W próbie na trzech grupach model bazowy był bezbłędny, a kandydat pogarszał macro-F1 o `0.2666666666666666`. Cały przedział różnicy był ujemny. Przy `minimum_effect=-1` otrzymano jednak `apply_candidate=True`.

Analogicznie `build_two_review_queues()` przyjął do kolejki poprawy modelu pozycję o oczekiwanym efekcie `-0.1` i dolnej granicy `-0.2`, gdy próg wynosił `-0.3`. W `auditing/utility_queue.py:95` istnieje już kontrola nieujemnego progu, więc równoległe interfejsy egzekwują różne zasady.

**Skutek:** niepoprawna konfiguracja może nadać pogorszeniu status dopuszczonej poprawy, wbrew kontraktowi funkcji. Domyślne progi równe zero nie wywołują tego przypadku; nie wykazano jego użycia w zamrożonych wynikach.

**Poprawka:** wymagać nieujemnego minimalnego globalnego zysku i zachować jawny warunek dodatniego efektu. Nie mylić tego z celowo ujemnym marginesem non-inferiority dla pojedynczej klasy. Ujednolicić walidację interfejsów i dodać test odrzucenia ujemnego progu.

### F06 · P2 · Ogólny OOF nie przenosi informacji o niezbieżnym dopasowaniu

**Lokalizacja:** `src/histo_audit/cross_validation/oof.py:500`, `src/histo_audit/cross_validation/oof.py:697`, `src/histo_audit/workflows/original_audit.py`.

Wbudowana regresja logistyczna zapisuje `converged_` z wyniku optymalizacji. `grouped_oof_predict()` po `fit()` od razu zapisuje predykcje, bez odczytania i utrwalenia tego statusu. W próbie z `max_iter=1` wszystkie trzy modele miały `converged_=False`, a funkcja zwróciła OOF z pełnym, jednokrotnym pokryciem próbek. Poprawny podział OOF nie oznacza poprawnego zakończenia optymalizacji.

Dodatkowo gałąź bez SciPy ustawia `converged_=True` po ustalonej liczbie kroków Adam bez sprawdzenia kryterium zbieżności (`oof.py:512`). To osobna słabość tego samego kontraktu.

**Skutek:** ogólny audyt może zwrócić normalnie wyglądający ranking bez informacji, że optymalizator nie osiągnął zadanej zbieżności. Nie oznacza to braku zbieżności zapisanych badań: m.in. PUMA i publiczna replikacja mają dodatkowe, odrębne kontrole.

**Poprawka:** zapisywać status i diagnostykę dopasowania osobno dla każdego foldu. Dla modelu, który udostępnia status, nie ignorować `False`; dla adaptera bez takiej informacji zapisać `unknown`, nie wymyślać sukcesu. Polityka wykonania powinna jawnie rozstrzygać, kiedy wynik ma być niedostępny, a kiedy dopuszczalny wyłącznie jako diagnostyczny. Dodać test braku zbieżności.

### F07 · P2 · Ułamkowe etykiety są po cichu zmieniane na inne klasy

**Lokalizacja:** `src/histo_audit/workflows/original_audit.py:136`.

`_validate_manifest()` sprawdza, czy etykiety są liczbami i czy `observed_label == pre_corruption_label`, po czym wykonuje `astype(np.int64)`. Manifest z etykietami `0.2` i `1.9` został przyjęty jako klasy `0` i `1`.

**Skutek:** błąd w CSV może zmienić znaczenie analizowanej klasy bez komunikatu. Nie zapisuje to zmienionych adnotacji do pliku źródłowego, lecz audyt pracuje już na innych wartościach niż dostarczone.

**Poprawka:** przed rzutowaniem wymagać wartości skończonych i całkowitych; potem sprawdzić przynależność do dozwolonych klas. `1.0` może odpowiadać klasie 1, ale `1.9` musi wywołać błąd. Podobną zasadę stosować na publicznych wejściach metryk i funkcji rankingowych, które rzutują etykiety.

### F08 · P2 · Funkcja sąsiedztwa ufa błędnej deklaracji foldów

**Lokalizacja:** `src/histo_audit/auditing/neighbours.py:227`, `tests/test_neighbours.py:168`.

`fold_safe_neighbour_disagreement()` wyznacza referencje z przekazanego `training_groups_by_fold`, lecz nie sprawdza, czy ta lista jest rozłączna z grupami ocenianego foldu. Nie sprawdza też, czy jedna grupa należy tylko do jednego foldu. Końcowe odfiltrowanie grupy samego zapytania jest słabszym warunkiem.

Próba z grupami `a,b,c,d`, foldami `[0,0,1,1]` i wszystkimi grupami zadeklarowanymi jako treningowe zwróciła pary sąsiadów `s0↔s1` oraz `s2↔s3`: sąsiadów z tego samego holdoutu. Standaryzacja również korzysta wtedy z błędnie dopuszczonych referencji. Istniejący test remisów sam używa grup rozdzielonych między foldy i wszystkich grup jako treningowych, więc tego naruszenia nie wychwytuje.

**Skutek:** luka walidacji na granicy API; błędna integracja może naruszyć deklarowaną separację. Standardowy przepływ przekazuje poprawną proweniencję OOF. Nie stwierdzono takiego wycieku w sprawdzonych opublikowanych artefaktach.

**Poprawka:** walidować przypisanie każdej grupy do jednego foldu oraz rozłączność referencji z całym holdoutem przed dopasowaniem skali i indeksu. Test remisów powinien zachować prawidłowe grupy; osobny test ma odrzucać nieprawidłową proweniencję.

### F09 · P2 · Instrukcja odtwarzania miesza odczyt dowodów z ponownym treningiem

**Lokalizacja:** `README.md:175`, `scripts/verify_aanca_selected_candidate.py:45`, `scripts/verify_aanca_selected_candidate.py:167`, `scripts/verify_puma_new_data_confirmation.py:60`, `scripts/verify_nucls_supervised_qc_feasibility.py:84`.

Sekcja „Recalculate released evidence” grupuje komendy o bardzo różnych wymaganiach. `verify_aanca_selected_candidate.py` wymaga surowego archiwum MoNuSAC, przygotowuje obrazy w dwóch skalach, korzysta z lokalnego dziennika wcześniejszego wyszukiwania i wywołuje `evaluate_full_nested()`, czyli ponownie dopasowuje modele. Potrzebna historia pod `artifacts/autoresearch/...` nie jest częścią zwykłego publicznego checkoutu.

Weryfikator PUMA nie trenuje ponownie 44 modeli, ale buduje manifest z surowych danych PUMA. Sprawdzenie NuCLS QC wymaga źródłowej bazy SQLite. Niektóre komendy domyślnie zapisują pliki weryfikacyjne lub raporty. Samo pobranie repozytorium i LFS nie wystarcza do wykonania całego bloku.

Podczas audytu uruchomienie weryfikatora wybranego kandydata przerwano po potwierdzeniu zakresu ponownego treningu. Nie zaliczono go jako PASS. Nie zmieniło to kanonicznego wyniku kandydata.

**Poprawka:** tabela dla każdej komendy: wymagane pliki, czy trenuje, czy zapisuje, zakres dowodów i orientacyjna kategoria kosztu. Pełne ponowne wykonanie kandydata przenieść do odpowiedniej sekcji. Jeżeli celem jest weryfikacja po publicznym checkoutcie, dodać odrębną ścieżkę opartą wyłącznie na udostępnionych dowodach i jasno opisać jej ograniczenia.

### F10 · P3 · README przedstawia kalibrację jako wykonany element standardowego rankingu

**Lokalizacja:** `README.md:126`, `src/histo_audit/auditing/calibration.py:138`.

README opisuje łączenie „calibrated label confidence” z sąsiedztwem. Istnieje poprawnie wydzielony moduł kalibracji, lecz wyszukiwanie wywołań `cross_fitted_temperature_calibration` znajduje definicję, eksport i testy, a nie zastosowanie w utrzymywanych ścieżkach oryginalnego audytu, PUMA i publicznej replikacji. Sam softmax oraz pomiar ECE/Brier nie dowodzą wykonania kalibracji.

**Poprawka:** opisać wykorzystywane wsparcie modelu dla etykiety z predykcji OOF. Kalibrację oznaczyć jako osobną możliwość wymagającą właściwych danych walidacyjnych i jawnego protokołu. Nie włączać jej po fakcie do zamrożonych wyników. `SPEC.md` już poprawnie traktuje kalibrację jako warunkową.

### F11 · P3 · Weryfikator wydania blokuje aktualizację daty bieżącego statusu

**Lokalizacja:** `scripts/verify_professor_release.py:194`.

Weryfikator wymaga obecności dokładnego tekstu `Updated: 1 September 2026` w STATUS.md. Podczas końcowej kontroli zmiana samej daty na datę tego audytu wywołała błąd weryfikacji. Aby zachować kontrakt istniejącego wydania bez zmiany kodu, przywrócono jego nagłówek i wyraźnie dopisano oddzielną datę aneksu z audytem. Nie ukryto tego nieudanego sprawdzenia ani nie zmieniono weryfikatora.

**Poprawka:** rozdzielić datę zamrożonego wydania od daty aktualizacji żywego dokumentu statusowego. Wiązać wydanie z jego autorytetem/manifestem, a poprawność bieżącej daty sprawdzać jako pole o określonym znaczeniu, zamiast wymagać przypadkowego fragmentu tekstu. To ograniczenie utrzymania dokumentacji, nie dowód wadliwości wyników.

## Zgodność działania, badań i opisu

| Element | Ocena |
|---|---|
| Ranking adnotacji do przeglądu eksperckiego; brak diagnozy | Zgodny w głównych ścieżkach i na stronie. |
| Niezmienność źródłowych adnotacji | Zachowana w sprawdzonych ścieżkach; restauracja dotyczy symulowanego eksperymentu i osobnych stanów etykiet. |
| Grupowe OOF i końcowy zbiór referencyjny | Silne kontrole w standardowym OOF i zapisanych protokołach; dodatkowe uszczelnienie sąsiedztwa opisuje F08. |
| Niezależność generatora korupcji i audytora | Jawna proweniencja i obsługa `circularity_risk`; nie zastępowano brakującego dowodu deklaracją sukcesu. |
| H4 | Liczby zgodne; nazwa i wyjaśnienie przedziału niezgodne, F01. |
| Brakujące H6 | Pokazane jako niedostępne, nie jako zerowy wynik; filtr tabeli zwrócił trzy takie wiersze. |
| PanNuke po odzyskiwaniu wykonania | Jawnie ograniczone do analizy eksploracyjnej po ujawnieniu wyników. |
| Identyczne realizacje części korupcji dla różnych seedów | Ograniczenie ujawnione; nie należy interpretować ich jako niezależnych replik biologicznych. |
| NuCLS: wcześniejszy wynik niekorzystny i osobny JP.1 | Rozdzielone w aktualnym pakiecie lokalnym; starsze wdrożenie nie zawiera JP.1. |
| RIVA i MIDOG++ | Obecne lokalnie i w zweryfikowanych artefaktach; nieobecne we wdrożeniu. |
| PUMA | Lokalny opis rozróżnia kontrolowany transfer, flagowanie/wykluczenie i brak naturalnej oceny ekspertów; zachowuje ograniczenia analiz po otwarciu wyników. |
| Kalibracja | Deklaracja w README jest silniejsza niż wykonany standardowy przepływ, F10. |
| „Jedna wersja modelu” | Repozytorium zawiera historyczne i współczesne przepływy; domyślny oryginalny audyt nie jest automatycznie wybranym kandydatem PUMA/RIVA. W instrukcji użycia trzeba wskazywać konkretny profil i konfigurację. |
| Strona jako produkt | Jest artykułem badawczym i prezentacją dowodów, a nie usługą do wysyłania nowych obrazów i otrzymywania audytu. To zgodne z jej obecną funkcją. |

Weryfikatory potwierdziły spójność zapisanych wartości w swoim zakresie. Nie zmienia to znaczenia punktów odniesienia: niezgodność ekspertów nie jest rozstrzygniętym błędem patologa, kontrolowana korupcja nie jest naturalnym błędem, a korzystny ranking nie dowodzi poprawy pracy klinicznej. Mała liczba grup w NuCLS pozostaje ograniczeniem badania, a nie usterką kodu.

## Co uprościć i zoptymalizować

### Zmiany uzasadnione teraz w V1

1. **Walidacja wspólna dla utrzymywanych interfejsów.** F04, F05 i F07 pokazują rzeczywisty rozjazd kontraktów. Niewielki zestaw walidatorów wejścia ograniczy powielanie i zmniejszy liczbę takich błędów. Nie łączyć z nim automatycznie niezależnych weryfikatorów dowodów, bo ich niezależność ma wartość kontrolną.
2. **Jawne metadane statystyki i treningu.** Typ przedziału, jednostka resamplingu i status optymalizatora powinny przechodzić od obliczenia do raportu. Dzięki temu prezentacja nie dopowiada znaczenia na podstawie samej nazwy `interval_95`.
3. **Jednoznaczne profile uruchomienia.** Oddzielić instrukcje historycznego benchmarku, oryginalnego audytu, zamrożonego wybranego kandydata i weryfikacji dowodów. To można zrobić bez zmiany algorytmów ani historycznych domyślnych parametrów.
4. **Weryfikacja wdrożenia po URL.** Lokalny pakiet jest poprawny według obecnych bramek, a publiczny stan mimo to się różni. Kontrola zdalnego HTML, JSON, wersji i manifestu daje więcej niż kolejna wyłącznie lokalna kontrola.
5. **Wdrożyć już wykonaną optymalizację grafik.** Sześć sprite'ów pobranych z hostingu ważyło łącznie 8 656 731 bajtów. Ich lokalne odpowiedniki zajmują 1 299 558 bajtów, około 85% mniej. To konkretna oszczędność przesyłanych danych dostępna bez nowego projektu grafiki. Nie jest to pomiar przyspieszenia strony o 85%.

### Redukcja architektury w V2

Największe pliki to m.in. `preregistration_amendment.py` — 9 721 linii, `run_tracking.py` — 5 441, `image_oof.py` — 4 688, `confirmatory_completion.py` — 4 507 i `pannuke/publication.py` — 4 422. Wiele z tej objętości obsługuje historyczne autorytety, odzyskiwanie wykonania, integralność i bramki. Nie jest to automatycznie kod martwy; część ma aktywne zależności także z nowszych funkcji.

Największy sens ma mniejszy rdzeń o wyraźnych granicach: dane i grupy → reprezentacje → OOF → ranking → polityka przeglądu → ocena → artefakt/raport. Stan wykonania, CLI i prezentacja powinny korzystać z tych granic zamiast rozszerzać wspólne wielotysięczne moduły.

Istniejąca decyzja D047 i fizyczne oddzielenie V2 są właściwym kierunkiem. Jej cel 15–30 tys. linii produkcyjnego Pythona jest założeniem architektonicznym, a nie obietnicą, że da się bezpiecznie usunąć określony procent obecnego V1. Samo przeniesienie funkcji między plikami zmniejszy rozmiar modułu, lecz nie zmniejszy złożoności procesu.

Nie rekomenduję usuwania testów naruszenia integralności, łączenia niezależnych weryfikatorów z kodem obliczeniowym, wyrzucania niekorzystnych wyników ani ponownego strojenia na otwartych zbiorach końcowych. Nie rekomenduję też zmiany modelu tylko dlatego, że repozytorium jest duże.

### Czas testów i wydajność

Pełny zestaw wykonał się w 20 min 37 s. Najwolniejsze pozycje dotyczyły odczytu, przebudowy i kontroli plikowych dowodów confirmatory, w tym testu trwającego 113 s. To kandydat do profilowania kosztu fixture'ów i ponownych odczytów, nie podstawa do usunięcia bramek. W czasie audytu wykonywano też inne zadania; ten pomiar nie stanowi porównywalnego benchmarku ani dowodu regresji.

Przed optymalizacją algorytmiczną potrzebny jest profil konkretnego, reprezentatywnego wykonania. Obecna implementacja sąsiedztwa już buduje indeks na fold i obsługuje zapytania partiami. Bez pomiaru nie ma uzasadnienia dla wymiany jej na metodę przybliżoną, która mogłaby zmienić ranking i zamrożone wyniki.

## Walidacja wykonana podczas audytu

| Sprawdzenie | Wynik i zakres |
|---|---|
| `uv run pytest --durations=20` | 1182 passed, 1 skipped, 1237.58 s. Pominięcie: semantyka zmiany nazwy otwartego pliku POSIX, nieobsługiwana w ten sam sposób na Windows. |
| `uv run ruff check .` | PASS. |
| `uv run ruff format --check .` | PASS; 224 pliki. |
| `uv run mypy src` | PASS; 105 sprawdzanych plików źródłowych. |
| `uv run histo-audit doctor` | PASS; dostępna CUDA i RTX 4070. |
| `uv run histo-audit experiment smoke --runs-root artifacts/qa/review-20261002-smoke` | PASS; pełny syntetyczny przepływ CLI. |
| `uv run python -I scripts/verify_professor_release.py` | PASS przed i po dopisaniu raportu; 19 autorytetów i 13 plików pakietu. Pośrednia zmiana daty STATUS.md ujawniła F11; końcowy PASS nastąpił po zachowaniu nagłówka wydania i oddzielnym datowaniu aneksu. |
| `uv run python -I scripts/present_demo.py --verify-only` | PASS; manifest lokalny `395cb4e4f2b057febbaea60f934b896380570a497d7b6435ca7accc22f23d514`. |
| `scripts/verify_primary_evidence.py` | PASS dla odzyskanego przebiegu primary i jego restauracji: 33 dostępne porównania, trzy niedostępne H6 oraz H4 odtworzone z zapisanych tablic. |
| `scripts/verify_nucls_external_validation.py` | PASS; zapisane wyniki, baseline'y i przedziały w zakresie weryfikatora. |
| `scripts/verify_monusac_external_validation.py` | PASS dla zapisanych dowodów. |
| `scripts/verify_puma_new_data_confirmation.py` | PASS; odczyt dowodów, siedem bramek, zapisane flagi zbieżności 44 modeli; bez ponownego treningu 44 modeli. |
| `scripts/verify_nucls_supervised_qc_feasibility.py` | PASS weryfikacji; sparowany naturalny endpoint pozostaje niedostępny. PASS nie znaczy, że uzyskano brakujące etykiety. |
| `scripts/verify_nucls_independent_pathologist_validation.py` | PASS; 11 artefaktów. |
| `scripts/run_public_pathologist_replication.py verify --dataset riva` | PASS. |
| `scripts/run_public_pathologist_replication.py verify --dataset midogpp` | PASS. |
| `scripts/verify_aanca_selected_candidate.py` | PRZERWANE; wymaga ponownego treningu i lokalnych danych. Nie zaliczone jako PASS. |
| Próby F04–F08 | Odtworzyły opisane zachowania na małych sztucznych danych. |
| Kontrola strony w Playwright | Lokalnie desktop 1440×900, mobile 390×844, reduced motion, czytelność bez JS, filtr H6 i puste wyszukiwanie. Brak zaobserwowanego poziomego overflow, brakujących obrazów i błędów konsoli. |
| Dostępne wdrożenie Hostinger | Otwiera się w przeglądarce; porównano tekst, JSON i wszystkie 12 plików z manifestu. Wyniki F02–F03. |

Dokładna komenda głównego odczytu primary:

```powershell
uv run python scripts/verify_primary_evidence.py artifacts/runs/20260727T133947.089370Z_pannuke_primary_orphan_recovery --restoration-directory artifacts/runs/20260727T133947.089370Z_pannuke_primary_orphan_recovery/restorations/primary_0027_8531672acd3c --json
```

Istniejące testy i weryfikatory przeszły, a nowe kontrprzykłady ujawniły luki poza ich pokryciem. Nie należy utożsamiać liczby zielonych testów z dowodem pełnej zgodności metodologicznej.

## Zalecana kolejność dalszych prac

1. Naprawić opis i metadane H4 oraz dodać test ich znaczenia. Nie zmieniać zamrożonych liczb.
2. Poprawić F04–F08 i dodać małe testy regresji, które odtwarzają konkretne błędy. Przejść wymagane testy, lint, format i odpowiedni CLI.
3. Doprecyzować instrukcje uruchomienia, dostępność danych i kalibrację, F09–F10; rozdzielić daty wydania i aktualizacji statusu, F11.
4. Zbudować i zweryfikować aktualny pakiet strony, przywrócić oficjalną domenę, wdrożyć kompletną publikację i porównać serwowane bajty z manifestem.
5. Rozwijać uproszczony rdzeń w oddzielnym V2. Nowe hipotezy, kalibrację i polityki oceniać na właściwie zamrożonych nowych danych, nie używać otwartych wyników V1 jako świeżego potwierdzenia.

Etapy pozostają bez zmian: naukowy `EXTERNAL_VALIDATION_COMPLETE`, prezentacyjny `DEMO_COMPLETE`. Odnotowana niedostępność strony i błędy opisu wymagają naprawy, ale ten audyt sam nie przepisuje historycznych etapów. Działanie dla nieprzejrzanych danych naturalnych pozostaje `retain_uncorrected`.
