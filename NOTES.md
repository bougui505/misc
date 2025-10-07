# Table of Contents

*   [LaTeX TikZ Tips](#latex-tikz-tips)
    *   [TikZ document with `tikz` on an empty A4 page](#tikz-document-with-tikz-on-an-empty-a4-page)
    *   [TikZ document with positioning](#tikz-document-with-positioning)
    *   [Minimal TikZ Picture Example Cropped](#minimal-tikz-picture-example-cropped)
    *   [TikZ nodes relative to page](#tikz-nodes-relative-to-page)
    *   [Define a custom color](#define-a-custom-color)
    *   [Underlining Text with the soul package](#underlining-text-with-the-soul-package)
    *   [Style definition in TikZ](#style-definition-in-tikz)
    *   [Compute the middle point between two nodes](#compute-the-middle-point-between-two-nodes)
    *   [Justified text in TikZ nodes](#justified-text-in-tikz-nodes)
*   [Apptainer Definition File Sections](#apptainer-definition-file-sections)
    *   [Header](#header)
    *   [Sections](#sections)
    *   [Extracting the Definition File from a SIF image](#extracting-the-definition-file-from-a-sif-image)
*   [Python Pip Requirements File](#python-pip-requirements-file)
    *   [Requirements File Syntax](#requirements-file-syntax)
    *   [Generating a `requirements.txt` file](#generating-a-requirementstxt-file)
    *   [Installing packages from a `requirements.txt` file](#installing-packages-from-a-requirementstxt-file)
*   [Linux Shell Tricks](#linux-shell-tricks)
    *   [Add a line with given content at a given line number](#add-a-line-with-given-content-at-a-given-line-number)
    *   [Print lines between two patterns](#print-lines-between-two-patterns)
    *   [Extract only one file from a tar.gz archive](#extract-only-one-file-from-a-tar.gz-archive)
*   [Neomutt](#neomutt)
    *   [Saving Read Messages](#saving-read-messages)
    *   [Toggle Headers](#toggle-headers)
*   [Makefile Notes](#makefile-notes)
    *   [Built-in Variables List](#built-in-variables-list)
    *   [Suppressing Command Output](#suppressing-command-output)
*   [Git Tips](#git-tips)
    *   [Clone a Specific Commit](#clone-a-specific-commit)

    # LaTeX TikZ Tips

    ## TikZ document with `tikz` on an empty A4 page

    A standard LaTeX document with TikZ diagrams, using the `article` document class and `tikz` package. `a4paper` specifies the page size, and you can add a default font size like `12pt`.

    ```latex
    \documentclass[a4paper,12pt]{article} % Specify A4 paper size and a default font size
    \usepackage{tikz}

    \begin{document}
    \pagestyle{empty} % No page numbers

    % Your TikZ diagrams and document content go here

    \end{document}
    ```

    ## TikZ document with positioning

    To use the `tikz` library and its `positioning` capabilities:

    ```latex
    \documentclass{article}
    \usepackage{tikz}
    \usetikzlibrary{positioning}

    \begin{document}

    \begin{tikzpicture}
        % Define the first node
        \node (A) {Node A};
        % Define a second node to the right of Node A, with a specified distance
        \node (B) [right=of A] {Node B};
        % Define a third node below Node A, with a specified distance
        \node (C) [below=of A] {Node C};
        % Define a fourth node below Node B, implicitly using the same distance as previous relative placements
        \node (D) [below=of B] {Node D};

        % Draw some connecting lines/arrows
        \draw[->] (A) -- (B);
        \draw[->] (A) -- (C);
        \draw[->] (B) -- (D);
        \draw[->] (C) -- (D);
    \end{tikzpicture}

    \end{document}
    ```

    ## Minimal TikZ Picture Example Cropped

    A minimal TikZ picture automatically cropped to its content, useful for standalone graphics, using the `standalone` document class with `tikz` option.

    ```latex
    \documentclass[tikz, border=2mm]{standalone}
    \begin{document}
    \begin{tikzpicture}
        \draw (0,0) circle (1cm);
        \node at (0,0) {Hello};
    \end{tikzpicture}
    \end{document}
    ```


    ## TikZ nodes relative to page

    Nodes can be placed relative to the page using the `current page` node with `remember picture, overlay` options for the `tikzpicture`. Anchors like `current page.south west` position elements on the physical page.

    ```latex
    \documentclass{article}
    \usepackage{tikz}
    \usetikzlibrary{calc} % Required for coordinate calculations

    \begin{document}
    \pagestyle{empty} % Optional: No page numbers for a cleaner example

    \begin{tikzpicture}[remember picture, overlay]
        % Place a node in the top left corner of the page (with some margin)
        \node at ($(current page.north west) + (1cm,-1cm)$) [anchor=north west] {Top Left Node};

        % Place a node in the bottom right corner of the page (with some margin)
        \node at ($(current page.south east) + (-1cm,1cm)$) [anchor=south east] {Bottom Right Node};

        % Place a node centered on the page
        \node at (current page.center) {Centered Node};
    \end{tikzpicture}

    Some text on the page to demonstrate the overlay.
    More text here.

    \end{document}
    ```

    ## Define a custom color

    Define custom colors in LaTeX using the `xcolor` package for consistent color palettes in TikZ diagrams.

    ```latex
    \documentclass{article}
    \usepackage{tikz}
    \usepackage{xcolor} % Required for \definecolor

    \definecolor{mygreen}{RGB}{34,139,34} % Define 'mygreen' using RGB values

    \begin{document}

    \begin{tikzpicture}
        % Draw a rectangle filled with the custom color
        \fill[mygreen] (0,0) rectangle (2,1);
        \node at (1,0.5) {Custom Color};

        % Draw a circle with a border in the custom color
        \draw[thick, mygreen] (3,0.5) circle (0.8cm);
    \end{tikzpicture}

    \end{document}
    ```

    ## Underlining Text with the soul package

    The `soul` package provides a robust `\ul` command for underlining. Unlike standard LaTeX `\underline`, `\ul` handles line breaks and kerning gracefully, suitable for underlining across multiple lines. Load the package and use `\ul{text}`.

    ```latex
    \documentclass{article}
    \usepackage{soul} % Required for \ul command

    \begin{document}

    Here is some \ul{underlined text}.

    This is a longer paragraph where text will be \ul{underlined, and the underlining
    will correctly break across multiple lines, demonstrating the advanced capabilities
    of the soul package in handling complex text layouts}.

    \end{document}
    ```

    ## Style definition in TikZ

    Define styles within a `tikzpicture` environment to apply consistent formatting to multiple nodes or paths. Styles encapsulate a set of drawing options, making diagrams cleaner and easier to modify.

    ```latex
    \documentclass[tikz, border=2mm]{standalone}
    \begin{document}
    \begin{tikzpicture}[
        % Define a style for 'my node'
        mynode/.style={
            draw=blue,
            thick,
            fill=blue!20,
            rounded corners,
            font=\sffamily\bfseries\Large % Added \Large for bigger font
        },
        % Define a style for 'my arrow'
        myarrow/.style={
            ->,
            thick,
            red
        }
    ]
        % Use the 'mynode' style for Node A
        \node[mynode] (A) at (0,0) {Start};
        % Use the 'mynode' style for Node B
        \node[mynode] (B) at (3,0) {Process};
        % Use the 'myarrow' style for the connection
        \draw[myarrow] (A) -- (B);
    \end{tikzpicture}
    \end{document}
    ```

    ## Compute the middle point between two nodes

    To compute the middle point between two existing nodes in TikZ, you can use the `calc` library. This library allows you to perform arithmetic operations on coordinates. The syntax `($(<node1>)!<fraction>!(<node2>)$)` calculates a point along the line connecting `<node1>` and `<node2>`. For the exact midpoint, use a `fraction` of `0.5`.

    First, ensure you load the `calc` library:
    `\usetikzlibrary{calc}`

    Then, you can place a node or draw from the midpoint:

    ```latex
    \documentclass[tikz, border=2mm]{standalone}
    \usetikzlibrary{positioning,calc} % Load positioning and calc libraries
    \begin{document}
    \begin{tikzpicture}
        % Define two nodes
        \node (nodeA) {Node A};
        \node (nodeB) [right=3cm of nodeA] {Node B};

        % Draw a line between them
        \draw (nodeA) -- (nodeB);

        % Place a new node exactly in the middle of nodeA and nodeB
        \node[fill=red!20, circle, inner sep=1pt] at ($(nodeA)!0.5!(nodeB)$) {M};

        % Alternatively, define a coordinate for the midpoint
        \coordinate (midpoint) at ($(nodeA)!0.5!(nodeB)$);

        % Now you can use 'midpoint' as any other coordinate
        % For example, draw an arrow from nodeA to the midpoint
        \draw[->] (nodeA) -- (midpoint);

        % Or draw an arrow from the midpoint to nodeB
        \draw[->] (midpoint) -- (nodeB);

    \end{tikzpicture}
    \end{document}
    ```

    ## Justified text in TikZ nodes

    To have justified text within a TikZ node, you need to specify a fixed `text width` for the node and then set the `align=justify` option. This will cause the text inside the node to stretch evenly to fill the specified width, similar to justified paragraphs in standard LaTeX.

    ```latex
    \documentclass[tikz, border=2mm]{standalone}
    \begin{document}
    \begin{tikzpicture}
        \node[
            draw,
            fill=blue!10,
            text width=5cm, % Set a fixed width for the text
            align=justify,  % Enable justified alignment
            font=\sffamily,
            inner sep=3mm    % Add some padding around the text
        ] at (0,0) {
            This is a paragraph of text that will be displayed
            within a TikZ node. By setting the 'text width' and
            'align=justify' options, the text will be
            justified, spreading evenly across the specified width.
            This is useful for creating professional-looking diagrams
            with longer textual explanations inside nodes.
        };

        \node[
            draw=red,
            fill=red!10,
            text width=3cm, % A narrower node
            align=justify,
            font=\sffamily\small,
            below=of current bounding box.south, % Place below the first node
            yshift=-1cm % Add some vertical space
        ] {
            A shorter example
            demonstrating text
            justification in a
            more compact node.
        };
    \end{tikzpicture}
    \end{document}
    ```

# Apptainer Definition File Sections

Apptainer definition files are composed of two primary parts: a Header and a series of Sections.

## Header

The Header describes the base OS for the container. It contains essential metadata such as the operating system to build from and the version of Apptainer that will be used.

For example:

```
BOOTSTRAP: docker
FROM: debian:13
```

## Sections

Sections are the main body of the definition file, each denoted by a `%` prefix. Here's a list of all possible sections:

*   **%setup**: This section is executed on the host system *outside* of the container, before the container image is built. It's typically used for preparing files or directories that will be copied into the container.
*   **%files**: This section lists files and directories from the host that should be copied into the container during the build process.
*   **%environment**: Variables defined here will be set at runtime inside the container.
*   **%post**: This section contains commands that are executed *inside* the container after the base OS has been set up. This is where you typically install software, set up configurations, and perform other build-time actions.
*   **%test**: Commands in this section are executed during the build process to verify the container's functionality. It also defines the default `apptainer test` command.
*   **%run**: This section defines the default command executed when the container is run with `apptainer run <image>`.
*   **%startscript**: This section defines the command executed when the container is run as an instance with `apptainer instance start <image> <instance_name>`.
*   **%labels**: This section allows you to define custom metadata labels (key-value pairs) for the container.
*   **%help**: This section contains markdown-formatted text that provides usage instructions or general information about the container. It's displayed when `apptainer help <image>` is run.
*   **%appinstall <app_name>**: Similar to `%post` but scoped to a specific application definition within the container.
*   **%applabels <app_name>**: Similar to `%labels` but scoped to a specific application definition.
*   **%apprun <app_name>**: Similar to `%run` but scoped to a specific application definition.
*   **%apptest <app_name>**: Similar to `%test` but scoped to a specific application definition.

## Extracting the Definition File from a SIF image

You can extract the definition file (the `.def` file) that was used to build a Apptainer/Singularity Image Format (SIF) file using the `apptainer inspect` command with the `--deffile` option. This is useful for reviewing how an existing image was constructed.

```bash
apptainer inspect --deffile <sif_file_path>
```

For example, to view the definition file of `my_image.sif`:

```bash
apptainer inspect --deffile my_image.sif
```

To save the definition file to a new file, you can redirect the output:

```bash
apptainer inspect --deffile my_image.sif > my_image.def
```

# Python Pip Requirements File

A `requirements.txt` file lists Python packages and their versions that a project depends on. This file ensures that everyone working on the project, and the deployment environment, uses the exact same versions of dependencies, preventing compatibility issues.

## Requirements File Syntax

Each line in a `requirements.txt` file typically specifies a single package and optionally its version.

Common formats include:

*   **Exact version**: `package_name==1.2.3`
    This specifies that exactly version `1.2.3` of `package_name` should be installed.

*   **Minimum version**: `package_name>=1.2.3`
    This specifies that version `1.2.3` or any newer version should be installed.

*   **Compatible release**: `package_name~=1.2.3`
    This specifies a version that is compatible with `1.2.3`, typically meaning `1.2.3`, `1.2.4`, but not `1.3.0` or `2.0.0`.

*   **Strictly less than version**: `package_name<1.2.3`
    This specifies that any version strictly less than `1.2.3` should be installed.

*   **Any version**: `package_name`
    If no version is specified, `pip` will install the latest available version.

*   **From a URL**: `package_name @ git+https://github.com/user/repo.git@branch_or_tag#egg=package_name`
    This allows installing a package directly from a version control system repository or a specific URL.

Comments can be added using the `#` symbol.

## Generating a `requirements.txt` file

To create a `requirements.txt` file from your current Python environment, use the `pip freeze` command:

```bash
pip freeze > requirements.txt
```

This command outputs all installed packages and their versions to standard output, which is then redirected into a file named `requirements.txt`.

## Installing packages from a `requirements.txt` file

To install all packages listed in a `requirements.txt` file into your current Python environment, use the `pip install` command with the `-r` (or `--requirement`) flag:

```bash
pip install -r requirements.txt
```

This command will read the specified file and install all packages listed within it.

# Linux Shell Tricks

## Add a line with given content at a given line number

To insert a new line with specific content at a particular line number within a file, you can use `sed`. This is useful for modifying configuration files or scripts non-interactively.

The `sed` command below inserts `YOUR_CONTENT` at `LINE_NUMBER` in `FILENAME`.

```bash
sed -i 'LINE_NUMBERi\YOUR_CONTENT' FILENAME
```

*   `-i`: This option edits the file in place. Without it, `sed` would print the result to standard output.
*   `LINE_NUMBERi\YOUR_CONTENT`: This is the `sed` command.
    *   `LINE_NUMBER`: The line number before which the new content will be inserted.
    *   `i`: The "insert" command.
    *   `\YOUR_CONTENT`: The actual content to be inserted. Note the backslash before the content, which is required. If `YOUR_CONTENT` contains slashes or other special `sed` characters, they may need to be escaped, or a different delimiter can be used (e.g., `'LINE_NUMBERi#YOUR_CONTENT#'`).

**Example:** To insert the line `Hello World` at line 3 of `myfile.txt`:

```bash
sed -i '3i\Hello World' myfile.txt
```

## Print lines between two patterns

To extract and print lines from a file that fall between two specific patterns (inclusive of the patterns themselves), you can use `sed` or `awk`.

**Using `sed`:**

The `sed` command below prints all lines from `FILENAME` starting from the line matching `PATTERN1` up to and including the line matching `PATTERN2`.

```bash
sed -n '/PATTERN1/,/PATTERN2/p' FILENAME
```

*   `-n`: Suppresses automatic printing of each line.
*   `/PATTERN1/,/PATTERN2/`: Specifies a range of lines from `PATTERN1` to `PATTERN2`.
*   `p`: Prints the lines within the specified range.

**Using `awk`:**

The `awk` command below achieves the same result, often providing more flexibility for complex logic.

```bash
awk '/PATTERN1/,/PATTERN2/' FILENAME
```

This simplified `awk` command works because when a range is specified (`/PATTERN1/,/PATTERN2/`), `awk` performs the default action (which is to print the line) for all lines within that range.

## Extract only one file from a tar.gz archive

To extract a single specific file from a `.tar.gz` (or `.tgz`) archive without extracting the entire contents, you can use the `tar` command with the `-x` (extract), `-z` (gzip), `-f` (file), and the path to the specific file.

```bash
tar -xzf <archive.tar.gz> <path/to/file/in/archive>
```

*   `-x`: Extract files from an archive.
*   `-z`: Filter the archive through `gzip` (for `.gz` archives).
*   `-f`: Specify the archive file name.

**Example:** To extract only `report.txt` from `data.tar.gz` located within a subdirectory `docs/`, you would use:

```bash
tar -xzf data.tar.gz docs/report.txt
```

If you want to extract the file to a different location or rename it during extraction, you can combine this with redirection or using the `--transform` option (for more advanced renaming), but for simple extraction, the above is sufficient. To extract to a different directory, navigate there first or specify the output directory with `-C`.

```bash
mkdir extracted_files
tar -xzf data.tar.gz -C extracted_files docs/report.txt
```

To extract a file to a specific output directory without keeping its full path from the archive, use the `--strip-components` option. This option *always* requires a numeric argument, specifying the number of leading directory components to remove from the file path during extraction. This is useful when the file you want is deeply nested but you only care about the file itself.

**Example:** To extract `report.txt` from `data.tar.gz` (where `report.txt` is located at `docs/reports/2023/report.txt` inside the archive) and place it directly into `output_dir` as `report.txt`:

```bash
mkdir output_dir
tar -xzf data.tar.gz docs/reports/2023/report.txt --strip-components=3 -C output_dir
```

Here, `--strip-components=3` removes `docs/`, `reports/`, and `2023/` from the path, leaving just `report.txt`.

**REMARKS**: according the tar man page:
```
       -C, --directory=DIR
              Change to DIR before performing any operations.  This option is order-sensitive, i.e. it affects all options that follow.
```
therefore, -C should be use with caution...

# Neomutt

## Saving Read Messages

Neomutt is a powerful, text-based email client known for its flexibility and extensive customization options. When managing emails, it's often useful to save messages that have already been read into a specific folder, for archiving or further processing.

To save all read messages in the current folder (e.g., your inbox) to another mailbox in Neomutt, you can use the following key sequence while in the index view:

1.  Press `T` (Tag) to enter tagging mode.
2.  Type `~R` to select all messages that have been read. This uses a pattern (`~R`) to match messages based on their read status.
3.  Press `;s` (semicolon) to apply a command to the tagged messages and `s` (save) to initiate the save action.

After pressing `s`, Neomutt will prompt you at the bottom of the screen to enter the name of the mailbox where the messages should be saved. For example, to save them to an `archive` folder, you would then type `archive` and press `Enter`.

## Toggle Headers

In Neomutt, you can toggle the visibility of message headers using a simple keybind. This is particularly useful for quickly decluttering the view and focusing on the message content itself.

To toggle between showing all headers and showing only essential headers:

1.  While viewing an email message, press `h`.

This action will cycle through different header display modes, typically from showing all headers to a minimal set, and back again.

# Makefile Notes

## Built-in Variables List

Make provides several automatic variables that are useful in recipes to refer to the target and prerequisites of the rule. These variables change their values for each rule.

Here are some commonly used built-in variables:

*   **`$@`**: The file name of the target of the rule. If the target is an archive member, then `$@` is the name of the archive file.
*   **`$<`**: The name of the first prerequisite. If the prerequisite is an archive member, then `$<` is the name of the member.
*   **`$^`**: The names of all the prerequisites, with spaces in between. For rules with pattern-specific prerequisites, this list does not contain any of the pattern-specific prerequisites.
*   **`$+`**: Similar to `$^`, but prerequisites listed more than once are duplicated in the list.
*   **`$?`**: The names of all the prerequisites that are newer than the target, with spaces in between. This variable is useful for rules that add dependencies to a list without rebuilding everything.
*   **`$*`**: The stem with which an implicit rule matches. If the target is `dir/a.foo.b`, and the target pattern is `dir/%.foo`, then the stem is `a.b`. The stem is useful for constructing the names of related files.
*   **`$(@D)`**: The directory part of `$@`. If `$@` is `dir/foo.o`, then `$(@D)` is `dir`. If `$@` does not contain a slash, `$(@D)` is `.`.
*   **`$(@F)`**: The file part of `$@`. If `$@` is `dir/foo.o`, then `$(@F)` is `foo.o`. `$(@F)` is equivalent to `$(notdir $@)`.
*   **`$(<D)`**: The directory part of `$<`. Similar to `$(@D)`.
*   **`$(<F)`**: The file part of `$<`. Similar to `$(@F)`.
*   **`$(basename $@)`**: The file name of the target without its extension. For example, if `$@` is `foo.txt`, then `$(basename $@)` is `foo`. If `$@` is `dir/bar.c`, then `$(basename $@)` is `dir/bar`.
*   **`$(basename $<)`**: The file name of the first prerequisite without its extension. Similar to `$(basename $@)`.

## Suppressing Command Output

By default, `make` prints each command line before executing it. This can make the output verbose, especially for simple commands. To prevent `make` from echoing a command, prefix the command with an `@` symbol.

This is useful for commands that produce their own descriptive output or for commands that you want to run silently.

**Example:**

Consider a Makefile with the following rule:

```makefile
greet:
	echo "Hello, world!"
	@echo "This line will not be echoed by Make."
```

When you run `make greet`:

*   The first `echo "Hello, world!"` command will be printed, followed by its output.
*   The second `@echo "This line will not be echoed by Make."` command will *not* be printed, only its output will appear.

**Output:**

```
echo "Hello, world!"
Hello, world!
This line will not be echoed by Make.
```

If you want to suppress output for all commands in a Makefile, you can add `.SILENT:` as a special target. This is equivalent to prefixing every command with `@`.

```makefile
.SILENT:

all:
	echo "This will also not be echoed."
```

# Git Tips

## Clone a Specific Commit

To clone a Git repository at a specific commit hash rather than the latest version, you can combine `git clone` with `git checkout`. This is useful for reproducing a build environment or examining the state of a project at a particular point in its history without cloning the entire repository's history depth.

First, perform a shallow clone of the repository to reduce download time and disk space. A depth of 1 is usually sufficient if you only need a single commit.

```bash
git clone <repository_url> --depth 1
```

Once the repository is cloned, navigate into its directory and use `git checkout` to switch to the desired commit hash. The `--force` (`-f`) flag might be necessary if you have any local changes, though typically not needed immediately after a fresh shallow clone.

```bash
cd <repository_name>
git checkout <commit_hash>
```

**Example:** To clone the `my-repo` repository from GitHub and then check out a specific commit `abcdef1234567890abcdef1234567890abcdef12`:

```bash
git clone https://github.com/user/my-repo.git --depth 1
cd my-repo
git checkout abcdef1234567890abcdef1234567890abcdef12
```
