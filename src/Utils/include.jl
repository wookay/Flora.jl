# module Flora.Utils

macro i(name)
    prefix = String(name)
    _include_file(prefix)
end

function _include_file(prefix::String)
    dir = pwd()
    for (root, dirs, files) in walkdir(dir)
        for file in files
            if startswith(file, prefix)
                Main.include(file)
                return
            end
        end
    end
end

# module Flora.Utils
