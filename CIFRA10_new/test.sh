str=./onnx
# shellcheck disable=SC2045
if [ -d "$str" ]; then for file in $(ls $str/)
	do
	  python test.py --name onnx/$file
	done
fi