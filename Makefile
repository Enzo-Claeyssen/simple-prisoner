
all :
	@make run


run :
	@echo ------
	@echo Running programm
	@echo ------
	@python3 -m venv env
	@. env/bin/activate; pip install pandas; pip install numpy; pip install matplotlib; pip install tqdm; python3 src
	@echo ------
	@echo Program ended
	@echo ------
