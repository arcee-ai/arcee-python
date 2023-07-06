.PHONY: style check_code_quality

export PYTHONPATH = .
check_dirs := arcee

style:
	black  $(check_dirs)

publish:
	rm -rf build dist
	python3 setup.py sdist bdist_wheel
	twine check dist/*
	twine upload dist/* -u ${PYPI_USERNAME} -p ${PYPI_PASSWORD} --verbose 