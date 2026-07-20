#include <netcdf.h>
#include <stddef.h>

int orch_nc_open_info(
    const char *path,
    int *iml,
    int *jml,
    int *lml,
    int *tml,
    int *fid
) {
    int dimid;
    size_t length;
    int status = nc_open(path, NC_NOWRITE, fid);
    if (status != NC_NOERR) return status;

    status = nc_inq_dimid(*fid, "longitude", &dimid);
    if (status != NC_NOERR) return status;
    status = nc_inq_dimlen(*fid, dimid, &length);
    if (status != NC_NOERR) return status;
    *iml = (int)length;

    status = nc_inq_dimid(*fid, "latitude", &dimid);
    if (status != NC_NOERR) return status;
    status = nc_inq_dimlen(*fid, dimid, &length);
    if (status != NC_NOERR) return status;
    *jml = (int)length;

    status = nc_inq_dimid(*fid, "level", &dimid);
    if (status != NC_NOERR) return status;
    status = nc_inq_dimlen(*fid, dimid, &length);
    if (status != NC_NOERR) return status;
    *lml = (int)length;

    status = nc_inq_dimid(*fid, "time", &dimid);
    if (status == NC_NOERR) {
        status = nc_inq_dimlen(*fid, dimid, &length);
        if (status != NC_NOERR) return status;
        *tml = (int)length;
    } else {
        *tml = 1;
    }
    return NC_NOERR;
}

int orch_nc_close(int fid) {
    return nc_close(fid);
}

int orch_nc_read_r1(int fid, const char *name, int n1, double *output) {
    int varid;
    int status = nc_inq_varid(fid, name, &varid);
    if (status != NC_NOERR) return status;
    for (int i = 0; i < n1; ++i) {
        size_t index[1] = {(size_t)i};
        status = nc_get_var1_double(fid, varid, index, &output[i]);
        if (status != NC_NOERR) return status;
    }
    return NC_NOERR;
}

int orch_nc_read_r2(
    int fid,
    const char *name,
    int n1,
    int n2,
    double *output
) {
    int varid;
    int status = nc_inq_varid(fid, name, &varid);
    if (status != NC_NOERR) return status;
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            size_t index[2] = {(size_t)i, (size_t)j};
            status = nc_get_var1_double(
                fid, varid, index, &output[i + n1 * j]
            );
            if (status != NC_NOERR) return status;
        }
    }
    return NC_NOERR;
}

int orch_nc_read_r3(
    int fid,
    const char *name,
    int n1,
    int n2,
    int n3,
    double *output
) {
    int varid;
    int status = nc_inq_varid(fid, name, &varid);
    if (status != NC_NOERR) return status;
    for (int i = 0; i < n1; ++i) {
        for (int j = 0; j < n2; ++j) {
            for (int k = 0; k < n3; ++k) {
                size_t index[3] = {
                    (size_t)i, (size_t)j, (size_t)k
                };
                status = nc_get_var1_double(
                    fid, varid, index, &output[i + n1 * (j + n2 * k)]
                );
                if (status != NC_NOERR) return status;
            }
        }
    }
    return NC_NOERR;
}
