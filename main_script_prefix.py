import dramatiq
from dramatiq.brokers.rabbitmq import RabbitmqBroker
from dramatiq.middleware import default_middleware
from dramatiq.results import Results
from dramatiq.results.backends import RedisBackend

dramatiq.set_encoder(dramatiq.PickleEncoder)
result_backend = RedisBackend()
middleware = [m() for m in default_middleware if m is not dramatiq.middleware.prometheus.Prometheus]
broker = RabbitmqBroker(middleware=middleware)

broker.add_middleware(Results(backend=result_backend))
dramatiq.set_broker(broker)
