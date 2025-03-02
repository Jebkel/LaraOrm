import asyncio
from collections import defaultdict
from datetime import datetime
from typing import TypeVar, Generic, Optional, List, Any, Type, Dict, Callable

import inflection
from aiomysql import Pool, DictCursor, create_pool
from loguru import logger
from pydantic import BaseModel, Field

from exceptions import EventCancelledError, RelationshipError, ORMError, NotFoundError

T = TypeVar('T', bound='Model')
R = TypeVar('R', bound='Model')


# Базовый класс для отношений
class Relation:
    def __init__(self, model: Type['Model'], foreign_key: str, local_key: str = 'id'):
        self.model = model
        self.foreign_key = foreign_key
        self.local_key = local_key


class HasMany(Relation):
    async def get(self, parent_id: Any) -> List['Model']:
        return await self.model.where(self.foreign_key, parent_id).get()


class BelongsTo(Relation):
    async def get(self, foreign_id: Any) -> Optional['Model']:
        return await self.model.where(self.local_key, foreign_id).first()


class QueryBuilder(Generic[T]):
    def __init__(self, model_cls: type[T]):
        self.model_cls = model_cls
        self._where = []
        self._params = {}
        self._limit = None
        self._order_by = []

    def order_by(self, column: str, direction: str = "ASC") -> 'QueryBuilder[T]':
        """Добавляет сортировку"""
        direction = direction.upper()
        if direction not in ("ASC", "DESC"):
            raise ValueError("Invalid sorting direction. Use ASC/DESC")
        self._order_by.append((column, direction))
        return self

    def where(self, column: str, operator: Any, value: Any = None) -> 'QueryBuilder[T]':
        param_name = f"param_{len(self._params)}"
        if value is None:
            value = operator
            operator = '='
        self._where.append(f"{column} {operator} %({param_name})s")
        self._params[param_name] = value
        return self

    def limit(self, count: int) -> 'QueryBuilder[T]':
        self._limit = count
        return self

    async def get(self) -> List[T]:
        query = f"SELECT * FROM {self.model_cls._table_name}"

        # Добавляем WHERE
        if self._where:
            query += " WHERE " + " AND ".join(self._where)

        # Добавляем ORDER BY
        if self._order_by:
            order_clause = ", ".join(
                [f"{col} {dir}" for col, dir in self._order_by]
            )
            query += f" ORDER BY {order_clause}"

        # Добавляем LIMIT
        if self._limit:
            query += f" LIMIT {self._limit}"

        logger.debug(f"Final query: {query}")
        async with self.model_cls.pool.acquire() as conn:
            async with conn.cursor(DictCursor) as cursor:
                await cursor.execute(query, self._params)
                result = await cursor.fetchall()
                return [self.model_cls(**row) for row in result]

    async def first(self) -> Optional[T]:
        self.limit(1)
        results = await self.get()
        return results[0] if results else None

    async def insert(self, data: Dict) -> T:
        # Фильтрация полей по модели
        model_fields = self.model_cls.model_fields
        filtered_data = {
            k: v for k, v in data.items()
            if k in model_fields and v is not None
        }

        columns = ", ".join(filtered_data.keys())
        placeholders = ", ".join([f"%({k})s" for k in filtered_data.keys()])

        query = (
            f"INSERT INTO {self.model_cls._table_name} "
            f"({columns}) VALUES ({placeholders})"
        )

        async with self.model_cls.pool.acquire() as conn:
            async with conn.cursor(DictCursor) as cursor:
                try:
                    await cursor.execute(query, filtered_data)  # Используем filtered_data
                    last_id = cursor.lastrowid
                    await conn.commit()

                    if last_id:
                        return await self.model_cls.where("id", last_id).first()
                    return None
                except Exception as e:
                    await conn.rollback()
                    logger.error(f"Insert error: {str(e)}")
                    raise

    async def update(self, **values) -> int:
        """Обновление записей по условиям"""
        if not values:
            raise ValueError("No values provided for update")

        set_clause = ", ".join([f"{k} = %({k})s" for k in values.keys()])
        query = f"UPDATE {self.model_cls._table_name} SET {set_clause}"

        if self._where:
            query += " WHERE " + " AND ".join(self._where)

        params = {**values, **self._params}

        async with self.model_cls.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                try:
                    await cursor.execute(query, params)
                    await conn.commit()
                    return cursor.rowcount
                except Exception as e:
                    await conn.rollback()
                    logger.error(f"Update error: {str(e)}")
                    raise

    async def refresh(self) -> None:
        if self.id is None:
            return
        updated = await self.__class__.where("id", self.id).first()
        self.__dict__.update(updated.__dict__)

        def with_relations(self, *relations: str) -> 'QueryBuilder[T]':
            self._relations = relations

        return self

    async def delete(self) -> int:

        """Массовое удаление записей по условиям"""
        query = f"DELETE FROM {self.model_cls._table_name}"

        if self._where:
            query += " WHERE " + " AND ".join(self._where)

        if self._limit:
            query += f" LIMIT {self._limit}"

        async with self.model_cls.pool.acquire() as conn:
            async with conn.cursor() as cursor:
                try:
                    await cursor.execute(query, self._params)
                    await conn.commit()
                    return cursor.rowcount
                except Exception as e:
                    await conn.rollback()
                    logger.error(f"Delete error: {str(e)}")
                    raise

    async def _load_relations(self, parents: List[T], relation_name: str):
        # Реализация жадной загрузки отношений
        if not parents:
            return

        first_parent = parents[0]
        relation = getattr(first_parent.__class__, relation_name, None)

        if not isinstance(relation, Relation):
            raise RelationshipError(f"Relation {relation_name} not found")

        ids = [getattr(p, relation.local_key) for p in parents]

        if isinstance(relation, HasMany):
            related = await relation.model.where(
                relation.foreign_key, ids
            ).get()

            related_map = {}
            for item in related:
                key = getattr(item, relation.foreign_key)
                if key not in related_map:
                    related_map[key] = []
                related_map[key].append(item)

            for parent in parents:
                setattr(parent, relation_name,
                        related_map.get(getattr(parent, relation.local_key), []))

        elif isinstance(relation, BelongsTo):
            foreign_ids = [getattr(p, relation.foreign_key) for p in parents]
            related = await relation.model.where(
                relation.local_key, foreign_ids
            ).get()

            related_map = {item.id: item for item in related}

            for parent in parents:
                setattr(parent, relation_name,
                        related_map.get(getattr(parent, relation.foreign_key)))


class Model(BaseModel):
    _table_name: Optional[str] = ""
    pool: Optional[Pool] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None

    def __init_subclass__(cls, **kwargs):
        if cls._table_name.default == "":
            cls._table_name = inflection.pluralize(cls.__name__.lower())

        # Инициализируем хранилища событий и отношений
        cls.__events = defaultdict(list)
        cls.__relations = {}

        super().__init_subclass__(**kwargs)

    @classmethod
    def _get_events(cls) -> defaultdict:
        return cls.__events

    @classmethod
    def where(cls, column: str, operator: Any, value: Any = None) -> QueryBuilder[T]:
        return QueryBuilder(cls).where(column, operator, value)

    @classmethod
    async def create_pool(cls, **kwargs) -> None:
        cls.pool = await create_pool(**kwargs)

    @classmethod
    async def create(cls: Type[T], **data) -> T:
        instance = cls(**data)
        return await instance.save()

    @classmethod
    async def find(cls, id: int) -> Optional[T]:
        return await cls.where("id", id).first()

    @classmethod
    async def find_or_fail(cls, id: int) -> T:
        result = await cls.find(id)
        if not result:
            raise NotFoundError(f"{cls.__name__} {id} not found")
        return result

    @classmethod
    def on(cls, event: str):
        def decorator(callback: Callable):
            cls._get_events()[event].append(callback)
            return callback

        return decorator

    async def _trigger_before(self, event: str) -> bool:
        allow = True
        for callback in self.__class__._get_events().get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    result = await callback(self)
                else:
                    result = callback(self)
                if result is False:
                    allow = False
            except Exception as e:
                logger.error(f"Event error in {event}: {str(e)}")
        return allow

    def _trigger_after(self, event: str):
        for callback in self.__class__._get_events().get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(self))
                else:
                    callback(self)
            except Exception as e:
                logger.error(f"Event error in {event}: {str(e)}")

    @classmethod
    def _get_relations(cls) -> dict:
        if not hasattr(cls, '__relations'):
            cls.__relations = {}
        return cls.__relations

    # Реализация отношений
    @classmethod
    def has_many(cls, model: Type['Model'], foreign_key: str):
        relation = HasMany(model, foreign_key)
        cls._get_relations()[model.__name__.lower()] = relation
        return relation

    @classmethod
    def belongs_to(cls, model: Type[R], foreign_key: str) -> BelongsTo:
        relation = BelongsTo(model, foreign_key)
        cls._relations[model.__name__.lower()] = relation
        return relation

    # Транзакции
    @classmethod
    async def transaction(cls):
        class TransactionContext:
            def __init__(self, pool):
                self.pool = pool
                self.conn = None

            async def __aenter__(self):
                self.conn = await self.pool.acquire()
                await self.conn.begin()
                return self.conn

            async def __aexit__(self, exc_type, exc, tb):
                if exc_type:
                    await self.conn.rollback()
                else:
                    await self.conn.commit()
                await self.pool.release(self.conn)

        return TransactionContext(cls.pool)

    # Метод delete
    async def delete(self) -> bool:
        if self.id is None:
            raise ORMError("Cannot delete instance without ID")
        allow = await self._trigger_before('deleting')
        if not allow:
            return False
        deleted_count = await self.__class__.where("id", self.id).delete()
        if deleted_count > 0:
            self._trigger_after('deleted')
            return True
        return False

    def __getattr__(self, name):
        if name in self._relations:
            relation = self._relations[name]
            if isinstance(relation, HasMany):
                return self._load_has_many(relation)
            elif isinstance(relation, BelongsTo):
                return self._load_belongs_to(relation)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    async def _load_has_many(self, relation: HasMany):
        key = getattr(self, relation.local_key)
        related = await relation.model.where(relation.foreign_key, key).get()
        setattr(self, relation.model.__name__.lower(), related)
        return related

    async def _load_belongs_to(self, relation: BelongsTo):
        key = getattr(self, relation.foreign_key)
        related = await relation.model.where(relation.local_key, key).first()
        setattr(self, relation.model.__name__.lower(), related)
        return related

    async def save(self) -> T:
        self.updated_at = datetime.now()
        if self.id is None:
            allow = await self._trigger_before('creating')
            if not allow:
                raise EventCancelledError("Creation cancelled by event listener")
            data = self.model_dump(exclude={'id'}, exclude_none=True)
            new_instance = await QueryBuilder(self.__class__).insert(data)
            self.id = new_instance.id
            self._trigger_after('created')
        else:
            allow = await self._trigger_before('updating')
            if not allow:
                raise ORMError("Update cancelled by event listener")
            data = self.model_dump(exclude={'id', 'created_at'}, exclude_none=True)
            query = f"UPDATE {self._table_name} SET {', '.join([f'{k}=%({k})s' for k in data.keys()])} WHERE id = %(id)s"
            params = {**data, 'id': self.id}
            async with self.__class__.pool.acquire() as conn:
                async with conn.cursor(DictCursor) as cursor:
                    await cursor.execute(query, params)
                    await conn.commit()
                    self._trigger_after('updated')
        return self

    class Config:
        arbitrary_types_allowed = True
