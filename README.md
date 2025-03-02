# AsyncORM Library

Асинхронная ORM библиотека для Python с поддержкой MySQL, построенная на базе `aiomysql` и `pydantic`.

# Установка

```
pip install git+https://github.com/Jebkel/LaraOrm
```

# Быстрый старт
## Определение моделей
```python
from datetime import datetime
from typing import Optional
from lara_orm import Model

class User(Model):
    id: Optional[int] = None
    name: str
    email: str
    created_at: datetime
```

# Инициализация подключения
```python
async def init_db():
    await Model.create_pool(
        host="localhost",
        port=3306,
        user="root",
        password="secret",
        db="mydatabase",
        autocommit=True
    )
```

# Основные возможности
## CRUD операции
### Создание записи
```python
user = await User.create(name="John", email="john@example.com")
```
### Поиск записей
```python
# Получить все записи
users = await User.where("rating", ">", 3).get()

# Оператор по умолчанию =
user = await User.where("rating", 3).get()

# Найти по ID
user = await User.find(1)

# Первая запись с сортировкой
user = await User.where("age", ">", 18).order_by("created_at", "DESC").first()
```
### Обновление
```python
user = await User.find(1)
user.name = "John Doe"
await user.save()

# Или массовое обновление
await User.where("status", "pending").update(status="approved")
```

### Удаление
```python
user = await User.find(1)
await user.delete()

# Массовое удаление
await User.where("expired", True).delete()
```

## Отношения
### 1:N Отношения
```python
class Post(Model):
    user_id: int
    title: str
    content: str
    
    @classmethod
    def belongs_to_user(cls):
        return cls.belongs_to(User, "user_id")

class User(Model):
    # ...
    
    @classmethod
    def has_posts(cls):
        return cls.has_many(Post, "user_id")
```

### Использование отношений
```python
# Получить все посты пользователя
posts = await user.posts.get()

# Получить автора поста
author = await post.user.get()
```

### Транзакции
```python
async with User.transaction() as conn:
    user = await User.create(name="Transaction Test")
    await user.delete()
    # Все операции будут отменены при исключении
```
### События
```python
# Вариант с исключением
@User.on("creating")
async def validate_user(user: User):
    if len(user.name) < 2:
        raise ValueError("Name too short")
    return True

# Вариант без исключения, с возвратом True или False (будет вызвано исключение в случае False)
@User.on("updating")
async def validate_updating_user(user: User):
    if len(user.name) < 2:
        return False

# События для действий, что уже произошли не должны вызывать исключений
@User.on("created")
def log_creation(user: User):
    print(f"New user created: {user.id}")

```

### Расширенные запросы
```python
# Комплексные условия
users = await User.where("age", ">", 18)
                 .where("verified", True)
                 .order_by("created_at")
                 .limit(10)
                 .get()

# Жадная загрузка
posts = await Post.where("rating", ">", 5).with_relations("user").get()
```

# API Reference
## Декораторы моделей
- ```@classmethod on(event: str)```  - регистрация обработчиков событий
- ```@classmethod has_many()```   - объявление отношения "один ко многим"
- ```@classmethod belongs_to()``` - объявление обратного отношения

### Основные методы

### Методы моделей
| Метод                            | Описание                                                                 |
|-----------------------------------|-------------------------------------------------------------------------|
| `Model.create(**data)`           | Создает новую запись в БД (асинхронный классовый метод)                 |
| `model.save()`                   | Сохраняет/обновляет объект в БД (асинхронный метод экземпляра)          |
| `model.delete()`                 | Удаляет запись из БД (асинхронный метод экземпляра)                     |
| `Model.find(id)`                 | Находит запись по ID (асинхронный классовый метод)                      |
| `Model.find_or_fail(id)`         | Находит запись или выбрасывает исключение (асинхронный классовый метод) |

### Методы построителя запросов
| Метод                            | Описание                                                                 |
|-----------------------------------|-------------------------------------------------------------------------|
| `.where(column, operator, value)`| Добавляет условие WHERE (`User.where("age", ">", 18)`)                  |
| `.order_by(column, direction)`   | Сортировка результатов (`order_by("created_at", "DESC")`)               |
| `.limit(count)`                  | Ограничивает количество результатов (`limit(10)`)                       |
| `.with_relations(*relations)`    | Жадная загрузка отношений (`with_relations("posts", "comments")`)       |
| `.get()`                         | Выполняет запрос и возвращает список объектов                          |
| `.first()`                       | Возвращает первый результат запроса                                    |
| `.update(**values)`              | Массовое обновление записей (`where("status", "new").update(status="processed")`) |
| `.delete()`                      | Массовое удаление записей (`where("expired", True).delete()`)          |

### Работа с отношениями
| Метод                            | Описание                                                                 |
|-----------------------------------|-------------------------------------------------------------------------|
| `Model.has_many(model, fk)`      | Объявляет отношение "один ко многим" (`User.has_many(Post, "user_id")`)|
| `Model.belongs_to(model, fk)`    | Объявляет отношение "принадлежит" (`Post.belongs_to(User, "user_id")`) |

### События
| Метод     | Описание                   |
|-----------|----------------------------|
| creating  | Перед созданием записи     |
| created   | После успешного создания   |
| updating  | Перед обновлением записи   |
| updated   | После успешного обновления |
| deleting  | Перед удалением записи     |
| deleted   | После успешного удаления   |


## Обработка ошибок
```python
try:
    user = await User.find_or_fail(999)
except NotFoundError as e:
    print(f"Ошибка: {e}")

try:
    await user.delete()
except EventCancelledError as e:
    print(f"Удаление отменено: {e}")
```