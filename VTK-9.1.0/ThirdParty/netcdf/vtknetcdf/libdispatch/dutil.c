/*********************************************************************
 *   Copyright 2018, UCAR/Unidata
 *   See netcdf/COPYRIGHT file for copying and redistribution conditions.
 *********************************************************************/

#include "config.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#ifdef HAVE_SYS_STAT_H
#include <sys/stat.h>
#endif
#ifdef HAVE_FCNTL_H
#include <fcntl.h>
#endif
#ifdef _MSC_VER
#include <io.h>
#endif
#include "netcdf.h"
#include "ncuri.h"
#include "ncbytes.h"
#include "nclist.h"
#include "nclog.h"
#include "ncpathmgr.h"

#define NC_MAX_PATH 4096

#define LBRACKET '['
#define RBRACKET ']'

/**************************************************/
/**
 * Provide a hidden interface to allow utilities
 * to check if a given path name is really an ncdap4 url.
 * If no, return null, else return basename of the url
 * minus any extension.
 */

int
NC__testurl(const char* path, char** basenamep)
{
    NCURI* uri;
    int ok = NC_NOERR;
    if(ncuriparse(path,&uri))
	ok = NC_EURL;
    else {
	char* slash = (uri->path == NULL ? NULL : strrchr(uri->path, '/'));
	char* dot;
	if(slash == NULL) slash = (char*)path; else slash++;
        slash = nulldup(slash);
        if(slash == NULL)
            dot = NULL;
        else
            dot = strrchr(slash, '.');
        if(dot != NULL &&  dot != slash) *dot = '\0';
        if(basenamep)
            *basenamep=slash;
        else if(slash)
            free(slash);
    }
    ncurifree(uri);
    return ok;
}

/* Return 1 if this machine is little endian */
int
NC_isLittleEndian(void)
{
    union {
        unsigned char bytes[SIZEOF_INT];
	int i;
    } u;
    u.i = 1;
    return (u.bytes[0] == 1 ? 1 : 0);
}

char*
NC_backslashEscape(const char* s)
{
    const char* p;
    char* q;
    size_t len;
    char* escaped = NULL;

    len = strlen(s);
    escaped = (char*)malloc(1+(2*len)); /* max is everychar is escaped */
    if(escaped == NULL) return NULL;
    for(p=s,q=escaped;*p;p++) {
        char c = *p;
        switch (c) {
	case '\\':
	case '/':
	case '.':
	case '@':
	    *q++ = '\\'; *q++ = '\\';
	    break;
	default: *q++ = c; break;
        }
    }
    *q = '\0';
    return escaped;
}

char*
NC_backslashUnescape(const char* esc)
{
    size_t len;
    char* s;
    const char* p;
    char* q;

    if(esc == NULL) return NULL;
    len = strlen(esc);
    s = (char*)malloc(len+1);
    if(s == NULL) return NULL;
    for(p=esc,q=s;*p;) {
	switch (*p) {
	case '\\':
	     p++;
	     /* fall thru */
	default: *q++ = *p++; break;
	}
    }
    *q = '\0';
    return s;
}

char*
NC_entityescape(const char* s)
{
    const char* p;
    char* q;
    size_t len;
    char* escaped = NULL;
    const char* entity;

    len = strlen(s);
    escaped = (char*)malloc(1+(6*len)); /* 6 = |&apos;| */
    if(escaped == NULL) return NULL;
    for(p=s,q=escaped;*p;p++) {
	char c = *p;
	switch (c) {
	case '&':  entity = "&amp;"; break;
	case '<':  entity = "&lt;"; break;
	case '>':  entity = "&gt;"; break;
	case '"':  entity = "&quot;"; break;
	case '\'': entity = "&apos;"; break;
	default	 : entity = NULL; break;
	}
	if(entity == NULL)
	    *q++ = c;
	else {
	    len = strlen(entity);
	    memcpy(q,entity,len);
	    q+=len;
	}
    }
    *q = '\0';
    return escaped;
}

char*
/*
Depending on the platform, the shell will sometimes
pass an escaped octotherpe character without removing
the backslash. So this function is appropriate to be called
on possible url paths to unescape such cases. See e.g. ncgen.
*/
NC_shellUnescape(const char* esc)
{
    size_t len;
    char* s;
    const char* p;
    char* q;

    if(esc == NULL) return NULL;
    len = strlen(esc);
    s = (char*)malloc(len+1);
    if(s == NULL) return NULL;
    for(p=esc,q=s;*p;) {
	switch (*p) {
	case '\\':
	     if(p[1] == '#')
	         p++;
	     /* fall thru */
	default: *q++ = *p++; break;
	}
    }
    *q = '\0';
    return s;
}

/**
Wrap mktmp and return the generated path,
or null if failed.
Base is the base file path. XXXXX is appended
to allow mktmp add its unique id.
Return the generated path.
*/

char*
NC_mktmp(const char* base)
{
    int fd = -1;
    char* tmp = NULL;
    size_t len;

    len = strlen(base)+6+1;
    if((tmp = (char*)malloc(len))==NULL)
        goto done;
    strncpy(tmp,base,len);
    strlcat(tmp, "XXXXXX", len);
    fd = NCmkstemp(tmp);
    if(fd < 0) {
       nclog(NCLOGERR, "Could not create temp file: %s",tmp);
       goto done;
    }
done:
    if(fd >= 0) close(fd);
    return tmp;
}

int
NC_readfile(const char* filename, NCbytes* content)
{
    int ret = NC_NOERR;
    FILE* stream = NULL;
    char part[1024];

#ifdef _WIN32
    stream = NCfopen(filename,"rb");
#else
    stream = NCfopen(filename,"r");
#endif
    if(stream == NULL) {ret=errno; goto done;}
    for(;;) {
	size_t count = fread(part, 1, sizeof(part), stream);
	if(count <= 0) break;
	ncbytesappendn(content,part,count);
	if(ferror(stream)) {ret = NC_EIO; goto done;}
	if(feof(stream)) break;
    }
    ncbytesnull(content);
done:
    if(stream) fclose(stream);
    return ret;
}

int
NC_writefile(const char* filename, size_t size, void* content)
{
    int ret = NC_NOERR;
    FILE* stream = NULL;
    void* p;
    size_t remain;

#ifdef _WIN32
    stream = NCfopen(filename,"wb");
#else
    stream = NCfopen(filename,"w");
#endif
    if(stream == NULL) {ret=errno; goto done;}
    p = content;
    remain = size;
    while(remain > 0) {
	size_t written = fwrite(p, 1, remain, stream);
	if(ferror(stream)) {ret = NC_EIO; goto done;}
	if(feof(stream)) break;
	remain -= written;
    }
done:
    if(stream) fclose(stream);
    return ret;
}

/*
Parse a path as a url and extract the modelist.
If the path is not a URL, then return a NULL list.
If a URL, but modelist is empty or does not exist,
then return empty list.
*/
int
NC_getmodelist(const char* path, NClist** modelistp)
{
    int stat=NC_NOERR;
    NClist* modelist = NULL;
    NCURI* uri = NULL;
    const char* modestr = NULL;
    const char* p = NULL;
    const char* endp = NULL;

    ncuriparse(path,&uri);
    if(uri == NULL) goto done; /* not a uri */

    /* Get the mode= arg from the fragment */
    modelist = nclistnew();    
    modestr = ncurifragmentlookup(uri,"mode");
    if(modestr == NULL || strlen(modestr) == 0) goto done;
    /* Parse the mode string at the commas or EOL */
    p = modestr;
    for(;;) {
	char* s;
	ptrdiff_t slen;
	endp = strchr(p,',');
	if(endp == NULL) endp = p + strlen(p);
	slen = (endp - p);
	if((s = malloc(slen+1)) == NULL) {stat = NC_ENOMEM; goto done;}
	memcpy(s,p,slen);
	s[slen] = '\0';
	nclistpush(modelist,s);
	if(*endp == '\0') break;
	p = endp+1;
    }

done:
    if(stat == NC_NOERR) {
	if(modelistp) {*modelistp = modelist; modelist = NULL;}
    }
    ncurifree(uri);
    nclistfree(modelist);
    return stat;
}

/*
Check "mode=" list for a path and return 1 if present, 0 otherwise.
*/
int
NC_testmode(const char* path, const char* tag)
{
    int stat = NC_NOERR;
    int found = 0;
    int i;
    NClist* modelist = NULL;

    if((stat = NC_getmodelist(path, &modelist))) goto done;
    for(i=0;i<nclistlength(modelist);i++) {
	const char* value = nclistget(modelist,i);
	if(strcasecmp(tag,value)==0) {found = 1; break;}
    }        
    
done:
    nclistfreeall(modelist);
    return found;
}

#ifdef __APPLE__
int isinf(double x)
{
    union { unsigned long long u; double f; } ieee754;
    ieee754.f = x;
    return ( (unsigned)(ieee754.u >> 32) & 0x7fffffff ) == 0x7ff00000 &&
           ( (unsigned)ieee754.u == 0 );
}

int isnan(double x)
{
    union { unsigned long long u; double f; } ieee754;
    ieee754.f = x;
    return ( (unsigned)(ieee754.u >> 32) & 0x7fffffff ) +
           ( (unsigned)ieee754.u != 0 ) > 0x7ff00000;
}

#endif /*APPLE*/

