# Материалы по курсу "Искусственные нейронные сети"

---

## Установка

### 0. Установка uv

Следуйте официальному руководству: [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/) или выполните команды ниже:

#### Linux / macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Windows

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

После установки проверьте версию:

```bash
uv --version
```

### 1. Клонирование репозитория

```bash
git clone https://github.com/malashinroman/ann_course.git
cd ann_course
```

### 2. Установка зависимостей

```bash
uv sync
```

---

> **Важно:** чтобы выполнять лабораторные работы, перейдите в соответствующую папку `labs/<номер_лабы>` и следуйте инструкциям в её файле `README`.
