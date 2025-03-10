# Add source directories
set(source_directories
        integration        
    )

foreach (dir IN LISTS source_directories)
    add_subdirectory(${dir})
endforeach ()
