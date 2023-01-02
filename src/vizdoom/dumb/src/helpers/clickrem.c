/*  _______         ____    __         ___    ___
 * \    _  \       \    /  \  /       \   \  /   /       '   '  '
 *  |  | \  \       |  |    ||         |   \/   |         .      .
 *  |  |  |  |      |  |    ||         ||\  /|  |
 *  |  |  |  |      |  |    ||         || \/ |  |         '  '  '
 *  |  |  |  |      |  |    ||         ||    |  |         .      .
 *  |  |_/  /        \  \__//          ||    |  |
 * /_______/ynamic    \____/niversal  /__\  /____\usic   /|  .  . ibliotheque
 *                                                      /  \
 *                                                     / .  \
 * clickrem.c - Click removal helpers.                / / \  \
 *                                                   | <  /   \_
 * By entheh.                                        |  \/ /\   /
 *                                                    \_  /  > /
 *                                                      | \ / /
 *                                                      |  ' /
 *                                                       \__/
 */

#include <stdlib.h>
#include <math.h>
#include "dumb.h"



typedef struct DUMB_CLICK DUMB_CLICK;


struct DUMB_CLICK_REMOVER
{
	DUMB_CLICK *click;
	int n_clicks;

	int offset;

	DUMB_CLICK *free_clicks;
};


struct DUMB_CLICK
{
	DUMB_CLICK *next;
	int32 pos;
	sample_t step;
};


static DUMB_CLICK *alloc_click(DUMB_CLICK_REMOVER *cr)
{
	if (cr->free_clicks != NULL)
	{
		DUMB_CLICK *click = cr->free_clicks;
		cr->free_clicks = click->next;
		return click;
	}
	return malloc(sizeof(DUMB_CLICK));
}

static void free_click(DUMB_CLICK_REMOVER *cr, DUMB_CLICK *cl)
{
	cl->next = cr->free_clicks;
	cr->free_clicks = cl;
}

DUMB_CLICK_REMOVER *DUMBEXPORT dumb_create_click_remover(void)
{
	DUMB_CLICK_REMOVER *cr = malloc(sizeof(*cr));
	if (!cr) return NULL;

	cr->click = NULL;
	cr->n_clicks = 0;

	cr->offset = 0;
	cr->free_clicks = NULL;

	return cr;
}



void DUMBEXPORT dumb_record_click(DUMB_CLICK_REMOVER *cr, int32 pos, sample_t step)
{
	DUMB_CLICK *click;

	ASSERT(pos >= 0);

	if (!cr || !step) return;

	if (pos == 0) {
		cr->offset -= step;
		return;
	}

	click = alloc_click(cr);
	if (!click) return;

	click->pos = pos;
	click->step = step;

	click->next = cr->click;
	cr->click = click;
	cr->n_clicks++;
}



static DUMB_CLICK *dumb_click_mergesort(DUMB_CLICK *click, int n_clicks)
{
	int i;
	DUMB_CLICK *c1, *c2, **cp;

	if (n_clicks <= 1) return click;

	/* Split the list into two */
	c1 = click;
	cp = &c1;
	for (i = 0; i < n_clicks; i += 2) cp = &(*cp)->next;
	c2 = *cp;
	*cp = NULL;

	/* Sort the sublists */
	c1 = dumb_click_mergesort(c1, (n_clicks + 1) >> 1);
	c2 = dumb_click_mergesort(c2, n_clicks >> 1);

	/* Merge them */
	cp = &click;
	while (c1 && c2) {
		if (c1->pos > c2->pos) {
			*cp = c2;
			c2 = c2->next;
		} else {
			*cp = c1;
			c1 = c1->next;
		}
		cp = &(*cp)->next;
	}
	if (c2)
		*cp = c2;
	else
		*cp = c1;

	return click;
}



void DUMBEXPORT dumb_remove_clicks(DUMB_CLICK_REMOVER *cr, sample_t *samples, int32 length, int step, double halflife)
{
	DUMB_CLICK *click;
	int32 pos = 0;
	int offset;
	int factor;

	if (!cr) return;

	factor = (int)floor(pow(0.5, 1.0/halflife) * (1U << 31));

	click = dumb_click_mergesort(cr->click, cr->n_clicks);
	cr->click = NULL;
	cr->n_clicks = 0;

	length *= step;

	while (click) {
		DUMB_CLICK *next = click->next;
		int end = click->pos * step;
		ASSERT(end <= length);
		offset = cr->offset;
		if (offset < 0) {
			offset = -offset;
			while (pos < end) {
				samples[pos] -= offset;
				offset = (int)(((LONG_LONG)(offset << 1) * factor) >> 32);
				pos += step;
			}
			offset = -offset;
		} else {
			while (pos < end) {
				samples[pos] += offset;
				offset = (int)(((LONG_LONG)(offset << 1) * factor) >> 32);
				pos += step;
			}
		}
		cr->offset = offset - click->step;
		free_click(cr, click);
		click = next;
	}

	offset = cr->offset;
	if (offset < 0) {
		offset = -offset;
		while (pos < length) {
			samples[pos] -= offset;
			offset = (int)((LONG_LONG)(offset << 1) * factor >> 32);
			pos += step;
		}
		offset = -offset;
	} else {
		while (pos < length) {
			samples[pos] += offset;
			offset = (int)((LONG_LONG)(offset << 1) * factor >> 32);
			pos += step;
		}
	}
	cr->offset = offset;
}



sample_t DUMBEXPORT dumb_click_remover_get_offset(DUMB_CLICK_REMOVER *cr)
{
	return cr ? cr->offset : 0;
}



void DUMBEXPORT dumb_destroy_click_remover(DUMB_CLICK_REMOVER *cr)
{
	if (cr) {
		DUMB_CLICK *click = cr->click;
		while (click) {
			DUMB_CLICK *next = click->next;
			free(click);
			click = next;
		}
		click = cr->free_clicks;
		while (click) {
			DUMB_CLICK *next = click->next;
			free(click);
			click = next;
		}
		free(cr);
	}
}



DUMB_CLICK_REMOVER **DUMBEXPORT dumb_create_click_remover_array(int n)
{
	int i;
	DUMB_CLICK_REMOVER **cr;
	if (n <= 0) return NULL;
	cr = malloc(n * sizeof(*cr));
	if (!cr) return NULL;
	for (i = 0; i < n; i++) cr[i] = dumb_create_click_remover();
	return cr;
}



void DUMBEXPORT dumb_record_click_array(int n, DUMB_CLICK_REMOVER **cr, int32 pos, sample_t *step)
{
	if (cr) {
		int i;
		for (i = 0; i < n; i++)
			dumb_record_click(cr[i], pos, step[i]);
	}
}



void DUMBEXPORT dumb_record_click_negative_array(int n, DUMB_CLICK_REMOVER **cr, int32 pos, sample_t *step)
{
	if (cr) {
		int i;
		for (i = 0; i < n; i++)
			dumb_record_click(cr[i], pos, -step[i]);
	}
}



void DUMBEXPORT dumb_remove_clicks_array(int n, DUMB_CLICK_REMOVER **cr, sample_t **samples, int32 length, double halflife)
{
	if (cr) {
		int i;
		for (i = 0; i < n >> 1; i++) {
			dumb_remove_clicks(cr[i << 1], samples[i], length, 2, halflife);
			dumb_remove_clicks(cr[(i << 1) + 1], samples[i] + 1, length, 2, halflife);
		}
		if (n & 1)
			dumb_remove_clicks(cr[i << 1], samples[i], length, 1, halflife);
	}
}



void DUMBEXPORT dumb_click_remover_get_offset_array(int n, DUMB_CLICK_REMOVER **cr, sample_t *offset)
{
	if (cr) {
		int i;
		for (i = 0; i < n; i++)
			if (cr[i]) offset[i] += cr[i]->offset;
	}
}



void DUMBEXPORT dumb_destroy_click_remover_array(int n, DUMB_CLICK_REMOVER **cr)
{
	if (cr) {
		int i;
		for (i = 0; i < n; i++) dumb_destroy_click_remover(cr[i]);
		free(cr);
	}
}
