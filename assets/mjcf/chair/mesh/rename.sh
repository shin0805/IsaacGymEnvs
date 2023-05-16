#!/bin/bash

# 対象ディレクトリの指定（カレントディレクトリの場合は "." を使用）
directory="."


# ディレクトリが存在するかチェック
if [ -d "tmp" ]; then
		rm -r tmp
fi
mkdir tmp

# ファイル名に含まれるスペースを取り除く処理
for file in "$directory"/*; do
	    if [[ -f "$file" ]]; then
					  newname=$(echo "$file" | tr -d ' ')
						cp "$file" "tmp/$newname"
			fi
done

