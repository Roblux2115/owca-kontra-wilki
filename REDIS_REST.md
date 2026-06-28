
## Uruchomienie Redis

Najprosciej przez Docker:

```bash
docker run --name owca-redis -p 6379:6379 redis:7-alpine
```

Jesli kontener juz istnieje:

```bash
docker start owca-redis
```

## Uruchomienie REST API

```bash
pip install -r requirements.txt
uvicorn state_server:app --reload
```

API bedzie dostepne pod:

```text
http://127.0.0.1:8000
```

Podglad dokumentacji:

```text
http://127.0.0.1:8000/docs
```

## Przykladowe endpointy

```text
/docs                    -> dokumentacja API
/state                   -> aktualny stan wszystkich
/state/normal_generator  -> aktualny stan + 10 ostatnich eventów generatora
/state/boss_generator    -> aktualny stan + 10 ostatnich eventów generatora bossów
/state/normal            -> trening zwykłego modelu
/state/boss              -> trening boss modelu
```

## Optymalizacja hiperparametrow

Skrypt `hyperparam_search.py` testuje kilka konfiguracji PPO i zapisuje wyniki do `hyperparam_results.json`.

Szybki test:

```bash
python hyperparam_search.py --mode normal --timesteps 10000 --eval-episodes 5
```

Dla aren bossow:

```bash
python hyperparam_search.py --mode boss --timesteps 10000 --eval-episodes 5
```

Status optymalizacji mozna podejrzec w API:

```text
http://127.0.0.1:8000/state/normal_hyperparam_search
http://127.0.0.1:8000/state/boss_hyperparam_search
```
