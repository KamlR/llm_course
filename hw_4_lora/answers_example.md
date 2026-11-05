# Примеры для ручной оценки

Одинаковые обращения для всех моделей выбраны до генерации: по одному из случайных разных классов. Автоматическое совпадение категории не заменяет ручной разбор; комментарии нужно заполнить после просмотра.

## Base

### Пример 1 — test[48]

Обращение:

    Can I put my old card back into the system? I just found it/

Правильная категория: `card_linking`

Ответ модели:

    card_arrival

Совпадение категории: **нет**.

Ручной комментарий: _заполнить после просмотра_.

### Пример 2 — test[2966]

Обращение:

    Top up on my Google Pay isn't working.

Правильная категория: `apple_pay_or_google_pay`

Ответ модели:

    top_up_by_card_charge

Совпадение категории: **нет**.

Ручной комментарий: _заполнить после просмотра_.

### Пример 3 — test[194]

Обращение:

    $1 extra has been charged on my statement, why is that?

Правильная категория: `extra_charge_on_statement`

Ответ модели:

    exchange_charge

Совпадение категории: **нет**.

Ручной комментарий: _заполнить после просмотра_.

### Пример 4 — test[1125]

Обращение:

    My address has changed.

Правильная категория: `edit_personal_details`

Ответ модели:

    change_personal_details

Совпадение категории: **нет**.

Ручной комментарий: _заполнить после просмотра_.

### Пример 5 — test[1757]

Обращение:

    Are there any restrictions causing my transfer to be declined?

Правильная категория: `declined_transfer`

Ответ модели:

    declined_transfer

Совпадение категории: **да**.

Ручной комментарий: _заполнить после просмотра_.

## Full FT

### Пример 1 — test[48]

Обращение:

    Can I put my old card back into the system? I just found it/

Правильная категория: `card_linking`

Ответ модели:

    card_linking

Совпадение категории: **да**.

Ручной комментарий: _заполнить после просмотра_.

### Пример 2 — test[2966]

Обращение:

    Top up on my Google Pay isn't working.

Правильная категория: `apple_pay_or_google_pay`

Ответ модели:

    apple_pay_or_google_pay

Совпадение категории: **да**.

Ручной комментарий: _заполнить после просмотра_.

### Пример 3 — test[194]

Обращение:

    $1 extra has been charged on my statement, why is that?

Правильная категория: `extra_charge_on_statement`

Ответ модели:

    extra_charge_on_statement

Совпадение категории: **да**.

Ручной комментарий: _заполнить после просмотра_.

### Пример 4 — test[1125]

Обращение:

    My address has changed.

Правильная категория: `edit_personal_details`

Ответ модели:

    edit_personal_details

Совпадение категории: **да**.

Ручной комментарий: _заполнить после просмотра_.

### Пример 5 — test[1757]

Обращение:

    Are there any restrictions causing my transfer to be declined?

Правильная категория: `declined_transfer`

Ответ модели:

    declined_transfer

Совпадение категории: **да**.

Ручной комментарий: _заполнить после просмотра_.

## LoRA

### Пример 1 — test[48]

Обращение:

    Can I put my old card back into the system? I just found it/

Правильная категория: `card_linking`

Ответ модели:

    card_linking

Совпадение категории: **да**.

Ручной комментарий: _заполнить после просмотра_.

### Пример 2 — test[2966]

Обращение:

    Top up on my Google Pay isn't working.

Правильная категория: `apple_pay_or_google_pay`

Ответ модели:

    apple_pay_or_google_pay

Совпадение категории: **да**.

Ручной комментарий: _заполнить после просмотра_.

### Пример 3 — test[194]

Обращение:

    $1 extra has been charged on my statement, why is that?

Правильная категория: `extra_charge_on_statement`

Ответ модели:

    extra_charge_on_statement

Совпадение категории: **да**.

Ручной комментарий: _заполнить после просмотра_.

### Пример 4 — test[1125]

Обращение:

    My address has changed.

Правильная категория: `edit_personal_details`

Ответ модели:

    edit_personal_details

Совпадение категории: **да**.

Ручной комментарий: _заполнить после просмотра_.

### Пример 5 — test[1757]

Обращение:

    Are there any restrictions causing my transfer to be declined?

Правильная категория: `declined_transfer`

Ответ модели:

    declined_transfer

Совпадение категории: **да**.

Ручной комментарий: _заполнить после просмотра_.
