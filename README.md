# Прогнозирование оттока клиентов
Работа с сервером, осуществляется следующим образом:
Клиент отправляет POST запрос на адресс сервера (localhost:9000/predict), в котором передает один или несколько файлов в формате JSON или CSV.

Ответ от сервера приходит в виде multipart/form-data, где каждый файл соответствует корректно обработанному файлу присланному пользователем.

в файле Analysis.ipynb содержится пример работы с сервером, а также обучечение модели и тестирование сервера.

Библиотеки необходимые для проекта:
* aiohttp
* pandas
* scikit-learn
* joblib
Их можно установить выполнив команду:
`./install.sh`

Запуск сервера:
`./server/server.py`

По умолчанию сервер запускается на порту 9000.

Сервер предусматривает базовые возможности для конфигурирования через конфигурационный файл, рассположенный в папке config.

Поля конфигурационного файла:
* port - номер порта, на котором будет запущен сервер;
* route - путь, по которому нужно отправлять запрос на сервер;
* modeldataparh - путь, по которому хранятся необходимые для прогнозирования данные;
* loggfile - путь, по которому сохраняются логги, записываемые сервером.
